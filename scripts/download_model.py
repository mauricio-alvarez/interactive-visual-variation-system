import sys
from pathlib import Path

from diffusers import StableDiffusionImg2ImgPipeline
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.settings import settings


def main() -> None:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        settings.model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.save_pretrained(f"models/{settings.model_id.replace('/', '__')}")
    print(f"Downloaded {settings.model_id}")


if __name__ == "__main__":
    main()
