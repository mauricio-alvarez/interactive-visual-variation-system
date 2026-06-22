from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

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
        preserve_faces: bool = True,
    ) -> list[dict[str, Any]]:
        if mode == "demo":
            return self._generate_demo_variations(input_path, output_dir, base_seed, preserve_faces)
        if mode == "api":
            return self._generate_openai_variations(input_path, output_dir, base_seed, preserve_faces)
        return self._generate_diffusion_variations(input_path, output_dir, base_seed, preserve_faces)

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
        preserve_faces: bool,
    ) -> list[dict[str, Any]]:
        import torch

        pipe = self._load_pipeline()
        image = self._prepare_image(input_path)
        negative_prompt = (
            "low quality, blurry, distorted face, changed identity, deformed eyes, "
            "asymmetrical face, malformed mouth, plastic skin, artifacts, watermark, text, "
            "extra limbs, deformed geometry, oversaturated"
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
            generated = result.images[0].convert("RGB")
            face_count = 0
            if preserve_faces:
                generated, face_count = self._preserve_faces(image, generated)
            generated = self._studio_finish(generated)

            out_path = output_dir / f"variation_{idx}.png"
            generated.save(out_path)
            results.append(self._metadata(idx, out_path, seed, style, preserve_faces, face_count, "local-gpu"))
        return results

    def _generate_openai_variations(
        self,
        input_path: Path,
        output_dir: Path,
        base_seed: int,
        preserve_faces: bool,
    ) -> list[dict[str, Any]]:
        import base64
        import os

        import httpx

        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or VISGEN_OPENAI_API_KEY is required for API studio mode.")

        prepared = self._prepare_image(input_path)
        api_input = output_dir / "api_input.png"
        prepared.save(api_input)
        image_bytes = api_input.read_bytes()

        headers = {"Authorization": f"Bearer {api_key}"}
        results: list[dict[str, Any]] = []
        with httpx.Client(timeout=180) as client:
            for idx, style in enumerate(self.styles, start=1):
                seed = base_seed + style.seed_offset
                prompt = self._api_edit_prompt(style, idx)
                files = [("image[]", ("portrait.png", image_bytes, "image/png"))]
                data = {
                    "model": settings.openai_image_model,
                    "prompt": prompt,
                    "size": settings.openai_image_size,
                    "quality": settings.openai_image_quality,
                }
                response = client.post(
                    f"{settings.openai_base_url.rstrip('/')}/images/edits",
                    headers=headers,
                    data=data,
                    files=files,
                )
                if response.status_code >= 400:
                    raise RuntimeError(f"OpenAI image edit failed: {response.status_code} {response.text}")

                payload = response.json()
                data_items = payload.get("data") or []
                if not data_items or not data_items[0].get("b64_json"):
                    raise RuntimeError("OpenAI image edit returned no image data.")
                b64_json = data_items[0]["b64_json"]
                image_data = base64.b64decode(b64_json)
                out_path = output_dir / f"variation_{idx}.png"
                out_path.write_bytes(image_data)
                face_count = len(self._detect_faces(prepared)) if preserve_faces else 0
                results.append(self._metadata(idx, out_path, seed, style, preserve_faces, face_count, "openai-api"))

        return results

    def _generate_demo_variations(
        self,
        input_path: Path,
        output_dir: Path,
        base_seed: int,
        preserve_faces: bool,
    ) -> list[dict[str, Any]]:
        image = self._prepare_image(input_path)
        transforms = [
            lambda im: self._natural_lighting(im),
            lambda im: self._cinematic_tint(im),
            lambda im: self._studio_headshot(im),
            lambda im: self._editorial_polish(im),
            lambda im: self._soft_luxury_retouch(im),
        ]
        results: list[dict[str, Any]] = []
        for idx, (style, transform) in enumerate(zip(self.styles, transforms), start=1):
            out_path = output_dir / f"variation_{idx}.png"
            generated = transform(image.copy())
            face_count = 0
            if preserve_faces:
                generated, face_count = self._preserve_faces(image, generated)
            generated.save(out_path)
            results.append(
                self._metadata(
                    idx,
                    out_path,
                    base_seed + style.seed_offset,
                    style,
                    preserve_faces,
                    face_count,
                    "preview",
                )
            )
        return results

    def _api_edit_prompt(self, style: VariationStyle, idx: int) -> str:
        return (
            "Edit the uploaded portrait into a professional photography result. "
            f"Studio look {idx}: {style.label}. {style.prompt}. "
            "Make this output visibly distinct from the other studio looks through lighting, "
            "color grade, background treatment, and photographic mood. Preserve the person's "
            "identity, facial structure, eye shape, nose, mouth, age, and natural skin texture. "
            "Do not beautify by changing identity. Avoid warped eyes, altered teeth, plastic skin, "
            "extra facial features, or artificial-looking retouching."
        )

    def _natural_lighting(self, image: Image.Image) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(1.12)
        image = ImageEnhance.Contrast(image).enhance(1.06)
        return ImageEnhance.Color(image).enhance(1.04)

    def _cinematic_tint(self, image: Image.Image) -> Image.Image:
        overlay = Image.new("RGB", image.size, (18, 32, 46))
        blended = Image.blend(image, overlay, alpha=0.22)
        blended = ImageEnhance.Color(blended).enhance(0.92)
        return ImageEnhance.Contrast(blended).enhance(1.16)

    def _studio_headshot(self, image: Image.Image) -> Image.Image:
        image = ImageOps.autocontrast(image)
        image = ImageEnhance.Brightness(image).enhance(1.06)
        return image.filter(ImageFilter.UnsharpMask(radius=1.4, percent=125))

    def _editorial_polish(self, image: Image.Image) -> Image.Image:
        cool = Image.new("RGB", image.size, (30, 38, 52))
        image = Image.blend(image, cool, alpha=0.08)
        image = ImageEnhance.Color(image).enhance(1.24)
        image = ImageEnhance.Contrast(image).enhance(1.18)
        return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110))

    def _soft_luxury_retouch(self, image: Image.Image) -> Image.Image:
        softened = image.filter(ImageFilter.GaussianBlur(radius=0.55))
        image = Image.blend(image, softened, alpha=0.18)
        warm = Image.new("RGB", image.size, (244, 228, 206))
        image = Image.blend(image, warm, alpha=0.1)
        image = ImageEnhance.Brightness(image).enhance(1.1)
        return ImageEnhance.Contrast(image).enhance(1.04)

    def _studio_finish(self, image: Image.Image) -> Image.Image:
        image = ImageEnhance.Contrast(image).enhance(1.04)
        image = ImageEnhance.Sharpness(image).enhance(1.06)
        return image

    def _preserve_faces(self, source: Image.Image, generated: Image.Image) -> tuple[Image.Image, int]:
        boxes = self._detect_faces(source)
        if not boxes:
            return generated, 0

        result = generated.copy()
        for box in boxes:
            mask = self._face_mask(source.size, box)
            face_layer = Image.composite(source, result, mask)
            result = Image.blend(result, face_layer, alpha=0.64)
        return result, len(boxes)

    def _detect_faces(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        import cv2
        import numpy as np

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            return []

        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(72, 72))
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def _face_mask(self, size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
        width, height = size
        x, y, w, h = box
        pad_x = int(w * 0.22)
        pad_y = int(h * 0.18)
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(width, x + w + pad_x)
        bottom = min(height, y + h + pad_y)

        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((left, top, right, bottom), radius=max(24, w // 5), fill=190)
        return mask.filter(ImageFilter.GaussianBlur(radius=max(12, w // 10)))

    def _metadata(
        self,
        idx: int,
        path: Path,
        seed: int,
        style: VariationStyle,
        preserve_faces: bool,
        detected_faces: int,
        provider: str,
    ) -> dict[str, Any]:
        return {
            "id": idx,
            "image_path": str(path),
            "seed": seed,
            "label": style.label,
            "provider": provider,
            "prompt": style.prompt,
            "strength": style.strength,
            "guidance_scale": style.guidance_scale,
            "face_preservation": preserve_faces,
            "detected_faces": detected_faces,
        }
