from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or .env."""

    project_name: str = "AI Portrait Studio"
    model_id: str = "runwayml/stable-diffusion-v1-5"
    lora_path: str = ""
    device: str = "cuda"
    image_size: int = 512
    num_variations: int = 5
    output_dir: Path = PROJECT_ROOT / "outputs" / "sessions"
    examples_dir: Path = PROJECT_ROOT / "examples"
    config_path: Path = PROJECT_ROOT / "config" / "generation.yaml"
    disable_safety_checker: bool = True
    explainer_provider: str = "rule"
    local_llm_model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_timeout_seconds: int = 30
    api_provider: str = "openai"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_image_model: str = "gpt-image-2"
    openai_image_size: str = "1024x1024"
    openai_image_quality: str = "medium"
    modelslab_api_key: str = ""
    modelslab_base_url: str = "https://stablediffusionapi.com/api/v3"
    modelslab_model_id: str = "runwayml/stable-diffusion-v1-5"
    modelslab_scheduler: str = "DDIM"
    modelslab_t2i_url: str = "https://modelslab.com/api/v7/images/text-to-image"
    modelslab_aspect_ratio: str = "1:1"
    modelslab_resolution: str = "1K"
    huggingface_api_key: str = ""
    huggingface_base_url: str = "https://api-inference.huggingface.co/models"
    huggingface_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    modal_endpoint_url: str = ""
    modal_api_key: str = ""
    modal_timeout_seconds: int = 240
    modal_num_inference_steps: int = 4
    modal_guidance_scale: float = 0.0
    modal_strength: float = 0.45
    finetuned_model_id: str = "runwayml/stable-diffusion-v1-5"
    finetuned_lora_path: str = ""
    finetuned_lora_weight_name: str = ""
    finetuned_lora_scale: float = 0.85
    finetuned_allow_base: bool = False
    finetuned_steps: int = 32
    ip_adapter_enabled: bool = False
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight_name: str = "ip-adapter-full-face_sd15.bin"
    ip_adapter_scale: float = 0.45
    face_lock_blend: float = 0.72

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="VISGEN_",
        extra="ignore",
    )


settings = Settings()
