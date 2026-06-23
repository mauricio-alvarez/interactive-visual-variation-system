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
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_image_model: str = "gpt-image-2"
    openai_image_size: str = "1024x1024"
    openai_image_quality: str = "medium"
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
