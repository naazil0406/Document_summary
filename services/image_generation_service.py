"""
Image generation service — Amazon Nova Canvas transport.

This is the second model in the image pipeline (see services/llm_service.py's
BaseLLMService.generate_image_prompt for the first: Nova Lite turning a user
request + retrieved RAG context into one optimized text prompt).

Nova Canvas is invoked differently from the Nova text models
(BedrockLLMService in llm_service.py): it does not use the Converse API,
it uses bedrock-runtime.invoke_model with a Nova-Canvas-specific JSON body,
and it returns base64-encoded PNG image bytes rather than text.
"""

import base64
import json
import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


class NovaCanvasService:
    """Transport over AWS Bedrock's Nova Canvas image-generation model."""

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