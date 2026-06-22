import sys
from pathlib import Path

from diffusers import StableDiffusionImg2ImgPipeline
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.settings import settings


def main() -> None:
    models_dir = ROOT / "models"
    cache_dir = models_dir / "hf_cache"
    local_dir = models_dir / settings.model_id.replace("/", "__")
    models_dir.mkdir(exist_ok=True)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        settings.model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        cache_dir=cache_dir,
    )
    pipe.save_pretrained(local_dir)
    print(f"Downloaded {settings.model_id} to {local_dir}")


if __name__ == "__main__":
    main()
