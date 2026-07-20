"""
Configuration module for the RAG application.

Loads environment variables (via python-dotenv) and exposes a single
immutable `settings` object used across the application. Also configures
application-wide logging.
"""

import os
import logging
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logging for the entire application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass(frozen=True)
class Settings:
    # --- Paths ---
    PDF_FOLDER: str = os.getenv("PDF_FOLDER", "data/pdfs")

    # --- AWS S3 (optional: documents are mirrored to S3 on upload and
    # fetched back down from S3 into PDF_FOLDER as a local working cache.
    # If S3_BUCKET_NAME is unset, the app runs local-folder-only — no
    # boto3 calls are made and no AWS credentials are required.) ---
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "")
    S3_PREFIX: str = os.getenv("S3_PREFIX", "documents/")

    # --- Chunking (Semantic Chunker config) ---
    SEMANTIC_BUFFER_SIZE: int = int(os.getenv("SEMANTIC_BUFFER_SIZE", "1"))
    SEMANTIC_BREAKPOINT_TYPE: str = os.getenv("SEMANTIC_BREAKPOINT_TYPE", "percentile")
    SEMANTIC_BREAKPOINT_AMOUNT: float = float(os.getenv("SEMANTIC_BREAKPOINT_AMOUNT", "95"))
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # --- Document Chunking (Stage 1: headings / sections / paragraphs) ---
    DOC_CHUNK_HEADING_MAX_LENGTH: int = int(os.getenv("DOC_CHUNK_HEADING_MAX_LENGTH", "80"))
    DOC_CHUNK_MIN_PARAGRAPH_LENGTH: int = int(os.getenv("DOC_CHUNK_MIN_PARAGRAPH_LENGTH", "20"))
    USE_SEMANTIC_BOUNDARY_DETECTION: bool = os.getenv(
        "USE_SEMANTIC_BOUNDARY_DETECTION", "true"
    ).strip().lower() in ("1", "true", "yes", "on")

    # --- Embeddings ---
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    INDEX_BATCH_SIZE: int = int(os.getenv("INDEX_BATCH_SIZE", "8"))

    # --- Qdrant (local Docker / local binary only) ---
    # Production uses a Qdrant instance on this machine — NOT Qdrant Cloud.
    # Start with:  docker compose up -d qdrant
    QDRANT_LOCAL: bool = os.getenv("QDRANT_LOCAL", "true").strip().lower() in (
        "1", "true", "yes", "on"
    )
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "company_docs")

    # --- Retrieval & Re-ranking ---
    TOP_K: int = int(os.getenv("TOP_K", "40"))
    TOP_K_SUMMARY: int = int(os.getenv("TOP_K_SUMMARY", "40"))
    MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.05"))
    USE_RERANKER: bool = os.getenv("USE_RERANKER", "true").strip().lower() in (
        "1", "true", "yes", "on"
    )
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "10"))

    # --- LLM provider switch: "openrouter" (default) or "bedrock" ---
    # Bedrock reuses the AWS_* credentials/region already configured above
    # for S3 — one set of AWS credentials covers both.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
    BEDROCK_REGION: str = os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1"))

    # --- OpenRouter (Qwen 2.5 72B) ---
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")

    # --- Bedrock model IDs (used only when LLM_PROVIDER=bedrock) ---
    # Q&A + summary: Nova Micro — text-only, cheapest, lowest latency.
    BEDROCK_MODEL: str = os.getenv("BEDROCK_MODEL", "amazon.nova-micro-v1:0")
    # Presentation: Nova 2 Lite — needs a region-prefixed inference profile
    # ID for on-demand access from a US region (see bedrock_llm.py docstring).
    BEDROCK_PRESENTATION_MODEL: str = os.getenv("BEDROCK_PRESENTATION_MODEL", "us.amazon.nova-2-lite-v1:0")

    # Q&A answers: 1024 tokens gives 4-5 solid lines without cutting off
    OPENROUTER_MAX_TOKENS: int = int(os.getenv("OPENROUTER_MAX_TOKENS", "1024"))

    OPENROUTER_TEMPERATURE: float = float(os.getenv("OPENROUTER_TEMPERATURE", "0.1"))
    OPENROUTER_SITE_URL: str = os.getenv("OPENROUTER_SITE_URL", "")
    OPENROUTER_SITE_NAME: str = os.getenv("OPENROUTER_SITE_NAME", "")

    # Summaries: 2048 tokens gives room for 8-10 detailed lines + key points
    SUMMARY_MAX_TOKENS: int = int(os.getenv("SUMMARY_MAX_TOKENS", "2048"))

    # Training scripts: same model by default, but a much higher token cap
    # since a full multi-section narration script is long-form output.
    PRESENTATION_MODEL: str = os.getenv("PRESENTATION_MODEL", os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct"))
    PRESENTATION_MAX_TOKENS: int = int(os.getenv("PRESENTATION_MAX_TOKENS", "8192"))
    # Deliberately higher than OPENROUTER_TEMPERATURE (used for factual Q&A/summaries):
    # the training-script story needs to vary run to run, not converge on one answer.
    PRESENTATION_TEMPERATURE: float = float(os.getenv("PRESENTATION_TEMPERATURE", "0.9"))

    # Local folder every generated training script is saved to as a .txt file,
    # in addition to being shown/downloadable in the Streamlit UI.
    NARRATIVE_SCRIPTS_DIR: str = os.getenv("NARRATIVE_SCRIPTS_DIR", "Narrative_scripts")

    # Dual Video Script + Story generator (see prompts/video_story_dual_system.txt).
    # Kept in its own directory rather than NARRATIVE_SCRIPTS_DIR so the
    # legacy single-panel "Story - X.mp4"-style scripts and these new
    # {video, story} pairs never collide on the "_script.txt" filename
    # convention _list_saved_scripts() relies on.
    DUAL_SCRIPTS_DIR: str = os.getenv("DUAL_SCRIPTS_DIR", "Narrative_scripts/dual")

    # --- Image generation pipeline (Nova Lite prompt -> image renderer) ---
    # Model 1: Nova Lite turns the user's request + retrieved RAG chunks
    # into one optimized image-generation prompt (see
    # prompts/image_prompt_system.txt / image_prompt_user.txt and
    # services/llm_service.py's generate_image_prompt()). Always Bedrock —
    # this pipeline does not have an OpenRouter path.
    BEDROCK_IMAGE_PROMPT_MODEL: str = os.getenv("BEDROCK_IMAGE_PROMPT_MODEL", "amazon.nova-lite-v1:0")
    IMAGE_PROMPT_MAX_TOKENS: int = int(os.getenv("IMAGE_PROMPT_MAX_TOKENS", "512"))
    IMAGE_PROMPT_TEMPERATURE: float = float(os.getenv("IMAGE_PROMPT_TEMPERATURE", "0.3"))

    # --- Learning Content Generation Engine ---
    # Turns (Content Type + Topic) into one short original piece of learning
    # content (Recall Card, Flashcard, Scenario, Daily Quiz, etc. — see
    # prompts/content_generation_system.txt), using the same LLM_PROVIDER
    # switch as get_qa_llm()/get_summary_llm(), just with its own token cap
    # and a higher temperature so repeated generations on the same topic
    # don't converge on identical wording.
    CONTENT_MAX_TOKENS: int = int(os.getenv("CONTENT_MAX_TOKENS", "400"))
    CONTENT_TEMPERATURE: float = float(os.getenv("CONTENT_TEMPERATURE", "0.7"))

    # Renderer 2: turns that prompt into the actual image. Three
    # interchangeable providers, all exposed as `.generate_image(prompt)` in
    # services/image_generation_service.py — see get_image_gen_service() in
    # main.py for the switch.
    #   "huggingface" (default): FLUX.1-dev via Hugging Face Inference
    #     Providers. Requires HF_TOKEN (from huggingface.co/settings/tokens);
    #     hf-inference has a free tier.
    #   "pollinations": Pollinations AI — free, no API key or AWS account
    #     required. https://pollinations.ai
    #   "aws": Amazon Bedrock Nova Canvas — requires AWS credentials and
    #     Bedrock model access; kept as an opt-in alternative.
    IMAGE_PROVIDER: str = os.getenv("IMAGE_PROVIDER", "huggingface")

    # Hugging Face / FLUX.1-dev settings (used when IMAGE_PROVIDER == "huggingface")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    HF_FLUX_MODEL: str = os.getenv("HF_FLUX_MODEL", "black-forest-labs/FLUX.1-dev")
    # "auto" lets Hugging Face route to whichever backend (fal, replicate,
    # together, hf-inference, etc.) currently serves the model, instead of
    # us hardcoding one provider that might not have it warm.
    HF_INFERENCE_PROVIDER: str = os.getenv("HF_INFERENCE_PROVIDER", "auto")
    FLUX_NUM_INFERENCE_STEPS: int = int(os.getenv("FLUX_NUM_INFERENCE_STEPS", "30"))
    FLUX_GUIDANCE_SCALE: float = float(os.getenv("FLUX_GUIDANCE_SCALE", "3.5"))

    # Pollinations AI settings (used when IMAGE_PROVIDER == "pollinations")
    POLLINATIONS_MODEL: str = os.getenv("POLLINATIONS_MODEL", "flux")
    POLLINATIONS_BASE_URL: str = os.getenv("POLLINATIONS_BASE_URL", "https://image.pollinations.ai/prompt")

    # Nova Canvas settings (used when IMAGE_PROVIDER == "aws"). Uses
    # invoke_model, not converse — a different call shape from every other
    # Bedrock model here.
    BEDROCK_IMAGE_GEN_MODEL: str = os.getenv("BEDROCK_IMAGE_GEN_MODEL", "amazon.nova-canvas-v1:0")
    IMAGE_QUALITY: str = os.getenv("IMAGE_QUALITY", "standard")
    IMAGE_CFG_SCALE: float = float(os.getenv("IMAGE_CFG_SCALE", "8.0"))

    # Shared by both providers.
    IMAGE_WIDTH: int = int(os.getenv("IMAGE_WIDTH", "1024"))
    IMAGE_HEIGHT: int = int(os.getenv("IMAGE_HEIGHT", "1024"))

    # Local folder generated images are saved to, in addition to being
    # returned directly in the API response (mirrors NARRATIVE_SCRIPTS_DIR).
    GENERATED_IMAGES_DIR: str = os.getenv("GENERATED_IMAGES_DIR", "Generated_images")

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


def _normalize_qdrant_url(url: str, local_only: bool) -> str:
    """Force localhost when QDRANT_LOCAL=true (ignore cloud URLs)."""
    normalized = (url or "http://localhost:6333").strip().rstrip("/")
    if not local_only:
        return normalized
    lowered = normalized.lower()
    if "localhost" in lowered or "127.0.0.1" in lowered:
        return normalized
    return "http://localhost:6333"


_settings = Settings()
_settings = Settings(
    **_settings.__dict__
    | {
        "QDRANT_URL": _normalize_qdrant_url(_settings.QDRANT_URL, _settings.QDRANT_LOCAL),
        "QDRANT_API_KEY": "" if _settings.QDRANT_LOCAL else _settings.QDRANT_API_KEY,
    }
)
settings = _settings
setup_logging(settings.LOG_LEVEL)