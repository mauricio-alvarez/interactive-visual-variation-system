from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from app.core.settings import PROJECT_ROOT, settings


@dataclass(frozen=True)
class VariationStyle:
    label: str
    prompt: str
    strength: float
    guidance_scale: float
    seed_offset: int


class ImageVariationGenerator:
    """Generate exactly five visual variations from one input image."""

    def __init__(self):
        self._pipeline = None
        self.styles = self._load_styles()

    def _load_styles(self) -> list[VariationStyle]:
        raw = yaml.safe_load(settings.config_path.read_text(encoding="utf-8"))
        styles = [
            VariationStyle(
                label=item["label"],
                prompt=item["prompt"],
                strength=float(item["strength"]),
                guidance_scale=float(item["guidance_scale"]),
                seed_offset=int(item["seed_offset"]),
            )
            for item in raw["variations"]
        ]
        if len(styles) != settings.num_variations:
            raise ValueError(f"Expected {settings.num_variations} styles, found {len(styles)}.")
        return styles

    def generate(
        self,
        input_path: Path,
        output_dir: Path,
        base_seed: int = 4200,
        mode: str = "diffusion",
    ) -> list[dict[str, Any]]:
        if mode == "demo":
            return self._generate_demo_variations(input_path, output_dir, base_seed)
        return self._generate_diffusion_variations(input_path, output_dir, base_seed)

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from diffusers import StableDiffusionImg2ImgPipeline

        if settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

        dtype = torch.float16 if settings.device == "cuda" else torch.float32
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "use_safetensors": True,
        }
        if settings.disable_safety_checker:
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False

        local_model = PROJECT_ROOT / "models" / settings.model_id.replace("/", "__")
        model_source = str(local_model) if local_model.exists() else settings.model_id
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_source, **kwargs)
        pipe = pipe.to(settings.device)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

        if settings.lora_path:
            pipe.load_lora_weights(settings.lora_path)

        self._pipeline = pipe
        return pipe

    def _prepare_image(self, path: Path) -> Image.Image:
        image = Image.open(path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        return ImageOps.fit(
            image,
            (settings.image_size, settings.image_size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )

    def _generate_diffusion_variations(
        self,
        input_path: Path,
        output_dir: Path,
        base_seed: int,
    ) -> list[dict[str, Any]]:
        import torch

        pipe = self._load_pipeline()
        image = self._prepare_image(input_path)
        negative_prompt = (
            "low quality, blurry, distorted, artifacts, watermark, text, extra limbs, "
            "deformed geometry, oversaturated"
        )

        results: list[dict[str, Any]] = []
        for idx, style in enumerate(self.styles, start=1):
            seed = base_seed + style.seed_offset
            generator = torch.Generator(device=settings.device).manual_seed(seed)
            result = pipe(
                prompt=style.prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=style.strength,
                guidance_scale=style.guidance_scale,
                num_inference_steps=28,
                generator=generator,
            )
            out_path = output_dir / f"variation_{idx}.png"
            result.images[0].save(out_path)
            results.append(self._metadata(idx, out_path, seed, style))
        return results

    def _generate_demo_variations(
        self,
        input_path: Path,
        output_dir: Path,
        base_seed: int,
    ) -> list[dict[str, Any]]:
        image = self._prepare_image(input_path)
        transforms = [
            lambda im: ImageEnhance.Brightness(im).enhance(1.18),
            lambda im: ImageEnhance.Color(im).enhance(1.35),
            lambda im: ImageOps.autocontrast(im),
            lambda im: im.filter(ImageFilter.UnsharpMask(radius=2, percent=160)),
            lambda im: self._cinematic_tint(im),
        ]
        results: list[dict[str, Any]] = []
        for idx, (style, transform) in enumerate(zip(self.styles, transforms), start=1):
            out_path = output_dir / f"variation_{idx}.png"
            transform(image.copy()).save(out_path)
            results.append(self._metadata(idx, out_path, base_seed + style.seed_offset, style))
        return results

    def _cinematic_tint(self, image: Image.Image) -> Image.Image:
        overlay = Image.new("RGB", image.size, (18, 32, 46))
        blended = Image.blend(image, overlay, alpha=0.16)
        return ImageEnhance.Contrast(blended).enhance(1.08)

    def _metadata(self, idx: int, path: Path, seed: int, style: VariationStyle) -> dict[str, Any]:
        return {
            "id": idx,
            "image_path": str(path),
            "seed": seed,
            "label": style.label,
            "prompt": style.prompt,
            "strength": style.strength,
            "guidance_scale": style.guidance_scale,
        }
