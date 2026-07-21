"""
Image generation services — FLUX.1-dev via Hugging Face Inference Providers
(default), Pollinations AI (free option), Freepik Mystic (optional), and
Amazon Nova Canvas (Bedrock, optional) transports.

All four are the second model in the image pipeline (see
services/llm_service.py's BaseLLMService.generate_image_prompt for the
first: Nova Lite turning a user request + retrieved RAG context into one
optimized text prompt). All four classes expose the same
`generate_image(prompt, negative_prompt="") -> bytes` interface, so
main.py's get_image_gen_service() can hand back whichever one based on
settings.IMAGE_PROVIDER.

FLUX.1-dev (black-forest-labs/FLUX.1-dev) is an open-weights image model
accessed here through the official huggingface_hub InferenceClient with
provider="auto" — Hugging Face routes the request to whichever backend
(fal, replicate, together, hf-inference, etc.) currently serves the model,
rather than us hardcoding one provider that might not have it warm. Requires
a Hugging Face access token (HF_TOKEN) with "Make calls to Inference
Providers" permission, plus accepting the model's gated license at
https://huggingface.co/black-forest-labs/FLUX.1-dev. Has a free tier,
subject to Hugging Face's serverless rate limits.

Pollinations AI (https://pollinations.ai) is a free, no-API-key-required
image generation HTTP API. A GET request to
https://image.pollinations.ai/prompt/{url-encoded-prompt} (with a few query
params) returns raw image bytes directly — no auth, no AWS account, no
Bedrock model access request needed. Kept here as a no-signup fallback.

Freepik Mystic (https://docs.freepik.com/api-reference/mystic) is Freepik's
in-house text-to-image model, requires a Freepik API key
(freepik.com/api -> API Dashboard -> API key), and is asynchronous:
POST /v1/ai/mystic submits the prompt and returns a task_id immediately
(status "IN_PROGRESS"); the actual image only exists once polling
GET /v1/ai/mystic/{task_id} reports status "COMPLETED", at which point
`generated` contains a temporary signed CDN URL that must be downloaded
separately to get the actual image bytes.

Nova Canvas is invoked differently from the Nova text models
(BedrockLLMService in llm_service.py): it does not use the Converse API,
it uses bedrock-runtime.invoke_model with a Nova-Canvas-specific JSON body,
and it returns base64-encoded PNG image bytes rather than text. Kept as
the AWS-hosted option.
"""

import base64
import io
import json
import logging
import time
from math import gcd
from urllib.parse import quote

import requests

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


class HuggingFaceFluxService:
    """Transport over FLUX.1-dev via the official huggingface_hub
    InferenceClient, using provider="auto" so Hugging Face picks whichever
    backend currently serves the model.

    Requires a Hugging Face access token with "Make calls to Inference
    Providers" permission, and that the token's account has accepted the
    gated license on the model page. See
    https://huggingface.co/docs/inference-providers/en/tasks/text-to-image.

    guidance_scale and num_inference_steps are tuned above FLUX's typical
    "natural photo" defaults (was 3.5 / 30) because this pipeline mostly
    asks FLUX for infographics with explicit colors and short in-image
    text labels, not photorealistic scenes. A higher guidance_scale (7.5)
    makes FLUX follow literal prompt instructions (named contrast colors,
    layout, label wording) more closely, and more steps (45) sharpens
    fine edges like icon outlines and short text/numbers. This does not
    guarantee perfectly legible text -- FLUX.1-dev is not a dedicated
    text-rendering model -- but it measurably reduces washed-out
    low-contrast output and garbled short labels versus the old defaults.
    """

    def __init__(
        self,
        api_token: str,
        model: str = "black-forest-labs/FLUX.1-dev",
        provider: str = "auto",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 45,
        guidance_scale: float = 7.5,
    ):
        if not api_token:
            raise RuntimeError(
                "HF_TOKEN is not set — required when IMAGE_PROVIDER='huggingface'."
            )
        self.model = model
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        self.client = InferenceClient(provider=provider, api_key=api_token)

    def generate_image(self, prompt: str, negative_prompt: str = "") -> bytes:
        """Send the Nova-Lite-produced prompt to FLUX.1-dev and return raw
        image bytes (encoded from the PIL.Image the client returns).

        Raises RuntimeError on any Hugging Face API failure, rather than
        silently returning empty bytes.
        """
        kwargs = {
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        logger.info(
            "Sending prompt to FLUX.1-dev model '%s' (prompt=%d chars).",
            self.model, len(prompt),
        )

        try:
            image = self.client.text_to_image(prompt, **kwargs)
        except HfHubHTTPError as exc:
            raise RuntimeError(
                f"Hugging Face request failed for model '{self.model}': {exc}"
            ) from exc

        if image is None:
            raise RuntimeError("Hugging Face returned no image data.")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()



class PollinationsImageService:
    """Transport over the free Pollinations AI image-generation API.

    No API key or AWS credentials required. See https://pollinations.ai.
    """

    def __init__(
        self,
        model: str = "flux",
        base_url: str = "https://image.pollinations.ai/prompt",
        width: int = 1024,
        height: int = 1024,
        nologo: bool = True,
        timeout: int = 60,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.width = width
        self.height = height
        self.nologo = nologo
        self.timeout = timeout

    def generate_image(self, prompt: str, negative_prompt: str = "") -> bytes:
        """Send the Nova-Lite-produced prompt to Pollinations AI and return
        raw image bytes.

        Pollinations has no dedicated negative-prompt field, so if one is
        supplied it's folded into the main prompt text as a simple
        instruction. Raises RuntimeError on any HTTP failure or empty
        response, rather than silently returning empty bytes.
        """
        full_prompt = prompt
        if negative_prompt:
            full_prompt = f"{prompt}. Avoid: {negative_prompt}"

        url = f"{self.base_url}/{quote(full_prompt)}"
        params = {
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "nologo": str(self.nologo).lower(),
        }

        logger.info(
            "Sending prompt to Pollinations AI model '%s' (prompt=%d chars).",
            self.model, len(prompt),
        )

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Pollinations AI request failed for model '{self.model}': {exc}"
            ) from exc

        image_bytes = response.content
        if not image_bytes:
            raise RuntimeError("Pollinations AI returned no image data.")

        return image_bytes


class FreepikImageService:
    """Transport over Freepik's text-to-image APIs.

    Freepik hosts several distinct models behind broadly the same
    asynchronous shape (submit a prompt, get a task_id back, poll until
    "COMPLETED", download the resulting CDN URL) but different endpoints
    and request bodies:

      - "realism" (default) or "flexible": Mystic style presets, Freepik's
        own in-house model, via POST /v1/ai/mystic. Supports resolution
        and filter_nsfw controls.
      - "flux-dev": Freepik-hosted FLUX.1-dev, via
        POST /v1/ai/text-to-image/flux-dev.
      - "hyperflux": Freepik-hosted fast Flux variant, via
        POST /v1/ai/text-to-image/hyperflux.
      - "seedream-v4-5": ByteDance's Seedream 4.5, via
        POST /v1/ai/text-to-image/seedream-v4-5. Purpose-built for dense,
        legible in-image text (labels, headings, short body copy) -- worth
        switching to specifically when Mystic/Flux output is coming back
        with garbled/misspelled text, which is an inherent limitation of
        those models rather than a prompt-wording problem.

    Selected via the FREEPIK_MODEL setting -- one API key, one env var,
    switches both which model and which endpoint gets used. Every option
    still exposes the same generate_image(prompt, negative_prompt="") ->
    bytes interface as every other provider in this module.

    Requires a Freepik API key (freepik.com/api -> API Dashboard -> API
    key). See https://docs.freepik.com/api-reference.
    """

    # Aspect ratio enum shared across every Freepik text-to-image endpoint.
    _ASPECT_RATIOS = {
        (1, 1): "square_1_1",
        (4, 3): "classic_4_3",
        (3, 4): "traditional_3_4",
        (16, 9): "widescreen_16_9",
        (9, 16): "social_story_9_16",
    }

    # Non-Mystic models each get their own endpoint under text-to-image/.
    # Anything not listed here (including "realism"/"flexible") is treated
    # as a Mystic style preset and goes to /v1/ai/mystic instead.
    _FLUX_ENDPOINTS = {
        "flux-dev": "https://api.freepik.com/v1/ai/text-to-image/flux-dev",
        "hyperflux": "https://api.freepik.com/v1/ai/text-to-image/hyperflux",
        "seedream-v4-5": "https://api.freepik.com/v1/ai/text-to-image/seedream-v4-5",
    }
    _MYSTIC_URL = "https://api.freepik.com/v1/ai/mystic"

    def __init__(
        self,
        api_key: str,
        model: str = "realism",
        resolution: str = "2k",
        width: int = 1024,
        height: int = 1024,
        filter_nsfw: bool = True,
        poll_interval: float = 3.0,
        poll_timeout: float = 120.0,
    ):
        if not api_key:
            raise RuntimeError(
                "FREEPIK_API_KEY is not set — required when IMAGE_PROVIDER='freepik'."
            )
        self.api_key = api_key
        self.model = model
        self.resolution = resolution
        self.aspect_ratio = self._aspect_ratio_for(width, height)
        self.filter_nsfw = filter_nsfw
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout

        self.submit_url = self._FLUX_ENDPOINTS.get(model, self._MYSTIC_URL)
        self.is_mystic = self.submit_url == self._MYSTIC_URL

    def _aspect_ratio_for(self, width: int, height: int) -> str:
        divisor = gcd(width, height) or 1
        ratio = (width // divisor, height // divisor)
        return self._ASPECT_RATIOS.get(ratio, "square_1_1")

    def _build_body(self, full_prompt: str) -> dict:
        if self.is_mystic:
            return {
                "prompt": full_prompt,
                "model": self.model,
                "resolution": self.resolution,
                "aspect_ratio": self.aspect_ratio,
                "filter_nsfw": self.filter_nsfw,
            }
        # The Flux-family endpoints use a simpler schema -- no style
        # preset, resolution, or filter_nsfw fields.
        return {
            "prompt": full_prompt,
            "aspect_ratio": self.aspect_ratio,
        }

    def generate_image(self, prompt: str, negative_prompt: str = "") -> bytes:
        """Submit the prompt to whichever Freepik endpoint FREEPIK_MODEL
        selects, poll until the task completes, then download and return
        the raw generated image bytes.

        None of Freepik's text-to-image endpoints have a dedicated
        negative-prompt field, so if one is supplied it's folded into the
        main prompt text as a simple instruction (same approach as
        PollinationsImageService). Raises RuntimeError on any request
        failure, a FAILED task, or a timeout waiting for completion,
        rather than silently returning empty bytes.
        """
        full_prompt = prompt
        if negative_prompt:
            full_prompt = f"{prompt}. Avoid: {negative_prompt}"

        headers = {
            "Content-Type": "application/json",
            "x-freepik-api-key": self.api_key,
        }
        body = self._build_body(full_prompt)

        logger.info(
            "Submitting prompt to Freepik (model=%s, endpoint=%s, prompt=%d chars).",
            self.model, self.submit_url, len(prompt),
        )

        try:
            response = requests.post(self.submit_url, headers=headers, json=body, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Freepik submission failed for model '{self.model}': {exc}") from exc

        task = (response.json() or {}).get("data", {})
        task_id = task.get("task_id")
        if not task_id:
            raise RuntimeError(f"Freepik ({self.model}) did not return a task_id.")

        image_url = self._poll_until_complete(task_id, headers)

        try:
            image_response = requests.get(image_url, timeout=60)
            image_response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to download the generated image from Freepik: {exc}"
            ) from exc

        image_bytes = image_response.content
        if not image_bytes:
            raise RuntimeError("Freepik image download returned no data.")
        return image_bytes

    def _poll_until_complete(self, task_id: str, headers: dict) -> str:
        status_url = f"{self.submit_url}/{task_id}"
        deadline = time.monotonic() + self.poll_timeout

        while True:
            try:
                status_response = requests.get(status_url, headers=headers, timeout=30)
                status_response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(f"Freepik status check failed: {exc}") from exc

            data = (status_response.json() or {}).get("data", {})
            status = data.get("status")

            if status == "COMPLETED":
                generated = data.get("generated", [])
                if not generated:
                    raise RuntimeError(
                        f"Freepik task '{task_id}' completed but returned no image."
                    )
                return generated[0]

            if status == "FAILED":
                raise RuntimeError(f"Freepik task '{task_id}' failed: {data}")

            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Freepik task '{task_id}' did not complete within "
                    f"{self.poll_timeout:.0f}s (last status: {status!r})."
                )
            time.sleep(self.poll_interval)


# Backwards-compatible alias -- FreepikImageService replaces the earlier,
# Mystic-only FreepikMysticService with a single class covering both
# Mystic and Freepik's separate Flux endpoints (flux-dev, hyperflux).
FreepikMysticService = FreepikImageService


class NovaCanvasService:
    """Transport over AWS Bedrock's Nova Canvas image-generation model.

    Optional AWS-hosted alternative to HuggingFaceFluxService (default) and
    PollinationsImageService (free) above — requires AWS credentials and
    Bedrock model access. Selected via settings.IMAGE_PROVIDER == "aws".
    """

    def __init__(
        self,
        model: str = "amazon.nova-canvas-v1:0",
        region_name: str = "us-east-1",
        aws_access_key_id: str = "",
        aws_secret_access_key: str = "",
        width: int = 1024,
        height: int = 1024,
        quality: str = "standard",
        cfg_scale: float = 8.0,
        number_of_images: int = 1,
    ):
        self.model = model
        self.width = width
        self.height = height
        self.quality = quality
        self.cfg_scale = cfg_scale
        self.number_of_images = number_of_images

        client_kwargs = {"region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self.client = boto3.client("bedrock-runtime", **client_kwargs)

    def generate_image(self, prompt: str, negative_prompt: str = "") -> bytes:
        """Send the Nova-Lite-produced prompt to Nova Canvas and return raw
        PNG image bytes (decoded from the base64 response).

        Raises RuntimeError on any Bedrock/boto3 failure or if no image
        comes back, rather than silently returning empty bytes.
        """
        text_to_image_params = {"text": prompt}
        if negative_prompt:
            text_to_image_params["negativeText"] = negative_prompt

        body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": text_to_image_params,
            "imageGenerationConfig": {
                "numberOfImages": self.number_of_images,
                "quality": self.quality,
                "width": self.width,
                "height": self.height,
                "cfgScale": self.cfg_scale,
            },
        }

        logger.info(
            "Sending prompt to Nova Canvas model '%s' (prompt=%d chars).",
            self.model, len(prompt),
        )

        try:
            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(
                f"Bedrock invoke_model failed for Nova Canvas model '{self.model}': {exc}"
            ) from exc

        response_body = json.loads(response["body"].read())

        error_message = response_body.get("error")
        if error_message:
            raise RuntimeError(f"Nova Canvas returned an error: {error_message}")

        images = response_body.get("images", [])
        if not images:
            raise RuntimeError("Nova Canvas returned no images.")

        return base64.b64decode(images[0])