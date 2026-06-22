from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or .env."""

    project_name: str = "Interactive Visual Variation System"
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

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="VISGEN_",
        extra="ignore",
    )


settings = Settings()
