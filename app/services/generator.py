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
        self._finetuned_pipeline = None
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
        styles: list[VariationStyle] | None = None,
    ) -> list[dict[str, Any]]:
        if mode == "demo":
            return self._generate_demo_variations(input_path, output_dir, base_seed, preserve_faces, styles)
        if mode == "api":
            return self._generate_api_variations(input_path, output_dir, base_seed, preserve_faces, styles)
        if mode == "finetuned":
            return self._generate_finetuned_variations(input_path, output_dir, base_seed, preserve_faces, styles)
        return self._generate_diffusion_variations(input_path, output_dir, base_seed, preserve_faces, styles)

    def api_provider(self) -> str:
        provider = (settings.api_provider or "openai").strip().lower()
        return provider if provider in {"openai", "modelslab", "huggingface", "modal"} else "openai"

    def api_key_configured(self) -> bool:
        import os

        if self.api_provider() == "modelslab":
            return bool(settings.modelslab_api_key or os.getenv("MODELSLAB_API_KEY"))
        if self.api_provider() == "huggingface":
            return bool(settings.huggingface_api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        if self.api_provider() == "modal":
            return bool(
                (settings.modal_endpoint_url or "").strip()
                or os.getenv("VISGEN_MODAL_ENDPOINT_URL", "").strip()
                or os.getenv("MODAL_ENDPOINT_URL", "").strip()
            )
        return bool(settings.openai_api_key or os.getenv("OPENAI_API_KEY"))

    def _generate_api_variations(
        self,
        input_path: Path,
        output_dir: Path,
        base_seed: int,
        preserve_faces: bool,
        styles: list[VariationStyle] | None,
    ) -> list[dict[str, Any]]:
        if self.api_provider() == "modelslab":
            return self._generate_modelslab_variations(input_path, output_dir, base_seed, preserve_faces, styles)
        if self.api_provider() == "huggingface":
            return self._generate_huggingface_variations(input_path, output_dir, base_seed, preserve_faces, styles)
        if self.api_provider() == "modal":
            return self._generate_modal_variations(input_path, output_dir, base_seed, preserve_faces, styles)
        return self._generate_openai_variations(input_path, output_dir, base_seed, preserve_faces, styles)

    def _generate_modal_variations(
        self,
        input_path: Path,
        output_dir: Path,
        base_seed: int,
        preserve_faces: bool,
        styles: list[VariationStyle] | None,
    ) -> list[dict[str, Any]]:
        import base64
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os

        import httpx

        endpoint = (
            (settings.modal_endpoint_url or "").strip()
            or os.getenv("VISGEN_MODAL_ENDPOINT_URL", "").strip()
            or os.getenv("MODAL_ENDPOINT_URL", "").strip()
        )
        if not endpoint:
            raise RuntimeError("VISGEN_MODAL_ENDPOINT_URL or MODAL_ENDPOINT_URL is required for API studio mode.")

        modal_api_key = (
            (settings.modal_api_key or "").strip()
            or os.getenv("VISGEN_MODAL_API_KEY", "").strip()
            or os.getenv("MODAL_API_KEY", "").strip()
        )

        prepared = self._prepare_image(input_path)
        api_input = output_dir / "api_input.png"
        prepared.save(api_input)
        input_b64 = base64.b64encode(api_input.read_bytes()).decode("utf-8")
        width, height = self._parse_image_size(settings.openai_image_size)
        active_styles = styles or self.styles

        headers = {"Content-Type": "application/json"}
        if modal_api_key:
            headers["Authorization"] = f"Bearer {modal_api_key}"

        timeout = max(30, int(settings.modal_timeout_seconds))
        def send_modal_request(idx: int, style: VariationStyle, seed: int) -> tuple[int, VariationStyle, int, str]:
            payload = {
                "prompt": self._api_edit_prompt(style, idx),
                "input_image_base64": input_b64,
                "negative_prompt": (
                    "blurry, low quality, distorted face, changed identity, malformed eyes, "
                    "waxy skin, artifacts"
                ),
                "width": width,
                "height": height,
                "strength": max(0.05, min(0.95, float(settings.modal_strength))),
                "num_inference_steps": max(1, min(8, int(settings.modal_num_inference_steps))),
                "guidance_scale": max(0.0, min(10.0, float(settings.modal_guidance_scale))),
                "seed": int(seed),
            }
            with httpx.Client(timeout=timeout) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                if response.status_code >= 400:
                    raise RuntimeError(f"Modal generation failed for variation {idx}: {response.status_code} {response.text}")

                payload_json = response.json()
                image_b64 = str(payload_json.get("image_base64", "")).strip()
                if not image_b64:
                    raise RuntimeError(f"Modal endpoint returned no image_base64 for variation {idx}: {payload_json}")
                return idx, style, seed, image_b64

        max_workers = min(5, max(1, len(active_styles)))
        responses: dict[int, tuple[VariationStyle, int, str]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(send_modal_request, idx, style, base_seed + style.seed_offset): idx
                for idx, style in enumerate(active_styles, start=1)
            }
            for future in as_completed(future_map):
                idx, style, seed, image_b64 = future.result()
                responses[idx] = (style, seed, image_b64)

        results: list[dict[str, Any]] = []
        for idx in range(1, len(active_styles) + 1):
            style, seed, image_b64 = responses[idx]
            out_path = output_dir / f"variation_{idx}.png"
            out_path.write_bytes(base64.b64decode(image_b64))

            generated = Image.open(out_path).convert("RGB")
            face_count = 0
            if preserve_faces:
                generated, face_count = self._preserve_faces(prepared, generated)
            generated = self._studio_finish(generated)
            generated.save(out_path)

            results.append(self._metadata(idx, out_path, seed, style, preserve_faces, face_count, "modal-api"))

        return results

    def _generate_huggingface_variations(
        self,
        input_path: Path,
        output_dir: Path,
        base_seed: int,
        preserve_faces: bool,
        styles: list[VariationStyle] | None,
    ) -> list[dict[str, Any]]:
        import os

        import httpx

        api_key = (
            settings.huggingface_api_key
            or os.getenv("HF_TOKEN", "")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        )
        if not api_key:
            raise RuntimeError(
                "HF_TOKEN, HUGGINGFACEHUB_API_TOKEN, or VISGEN_HUGGINGFACE_API_KEY is required for API studio mode."
            )

        prepared = self._prepare_image(input_path)
        active_styles = styles or self.styles
        endpoint = f"{settings.huggingface_base_url.rstrip('/')}/{settings.huggingface_model_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "image/png",
            "Content-Type": "application/json",
        }

        results: list[dict[str, Any]] = []
        with httpx.Client(timeout=300) as client:
            for idx, style in enumerate(active_styles, start=1):
                seed = base_seed + style.seed_offset
                payload = {
                    "inputs": self._api_text_to_image_prompt(style, idx),
                    "parameters": {
                        "negative_prompt": (
                            "blurry, low quality, distorted face, changed identity, malformed eyes, "
                            "waxy skin, artifacts"
                        ),
                        "guidance_scale": max(4.0, float(style.guidance_scale)),
                        "num_inference_steps": 30,
                        "width": settings.image_size,
                        "height": settings.image_size,
                        "seed": int(seed),
                    },
                    "options": {"wait_for_model": True, "use_cache": False},
                }
                response = client.post(endpoint, headers=headers, json=payload)
                if response.status_code >= 400:
                    raise RuntimeError(f"Hugging Face generation failed: {response.status_code} {response.text}")

                content_type = (response.headers.get("content-type") or "").lower()
                if "image/" in content_type:
                    image_bytes = response.content
                else:
                    payload_json = response.json()
                    error_text = payload_json.get("error") if isinstance(payload_json, dict) else None
                    raise RuntimeError(
                        f"Hugging Face did not return an image. Response: {error_text or payload_json}"
                    )

                out_path = output_dir / f"variation_{idx}.png"
                out_path.write_bytes(image_bytes)

                generated = Image.open(out_path).convert("RGB")
                face_count = 0
                if preserve_faces:
                    generated, face_count = self._preserve_faces(prepared, generated)
                generated = self._studio_finish(generated)
                generated.save(out_path)

                results.append(
                    self._metadata(idx, out_path, seed, style, preserve_faces, face_count, "huggingface-api")
                )

        return results

    def refined_styles_from_feedback(
        self,
        decisions: list[dict[str, Any]],
        previous_variations: list[dict[str, Any]],
        preference_profile: dict[str, Any] | None = None,
    ) -> list[VariationStyle]:
        accepted_ids = {
            int(item.get("variation_id", 0))
            for item in decisions
            if item.get("decision") == "accepted"
        }
        if not accepted_ids:
            raise RuntimeError("Refinement requires at least one kept look.")

        accepted_variations = [item for item in previous_variations if int(item.get("id", 0)) in accepted_ids]
        if not accepted_variations:
            raise RuntimeError("Refinement could not match kept looks to the previous variation set.")

        profile = preference_profile or {}
        prompt_directive = str(profile.get("prompt_directive", "")).strip()
        negative_directive = str(profile.get("negative_directive", "")).strip()
        strength_shift = float(profile.get("strength_shift", 0.0))
        guidance_shift = float(profile.get("guidance_shift", 0.0))

        refined_styles: list[VariationStyle] = []
        for idx in range(settings.num_variations):
            base = accepted_variations[idx % len(accepted_variations)]
            source_label = str(base.get("label", "Studio look")).strip() or "Studio look"
            source_prompt = self._strip_refinement_boilerplate(str(base.get("prompt", "")).strip())
            source_strength = float(base.get("strength", 0.4))
            source_guidance = float(base.get("guidance_scale", 7.0))
            step = idx - (settings.num_variations // 2)
            prompt_parts = [
                source_prompt,
                "Refine from approved feedback while preserving identity and realistic skin.",
            ]
            if prompt_directive:
                prompt_parts.append(self._truncate_prompt_directive(prompt_directive))
            if negative_directive:
                prompt_parts.append(self._truncate_prompt_directive(negative_directive))

            refined_styles.append(
                VariationStyle(
                    label=f"{source_label} refined {idx + 1}",
                    prompt=" ".join(prompt_parts),
                    strength=max(0.2, min(0.62, source_strength + strength_shift + (0.02 * step))),
                    guidance_scale=max(5.8, min(9.2, source_guidance + guidance_shift + (0.35 * step))),
                    seed_offset=101 + (idx * 17),
                )
            )
        return refined_styles

    def _strip_refinement_boilerplate(self, prompt: str) -> str:
        markers = [
            "Refine this look from approved user feedback.",
            "Refine from approved feedback while preserving identity and realistic skin.",
            "Keep the same subject identity and push professional portrait quality, better facial geometry, cleaner skin texture, and lighting consistency.",
            "Emphasize",
            "Avoid",
        ]
        for marker in markers:
            if marker in prompt:
                prompt = prompt.split(marker, 1)[0].strip()
        return prompt.rstrip(" .")

    def _truncate_prompt_directive(self, text: str, max_words: int = 8) -> str:
        words = text.strip().split()
        if len(words) <= max_words:
            return text.strip()
        return " ".join(words[:max_words]).rstrip(";,.") + "."

    def finetuned_configured(self) -> bool:
        return bool(settings.finetuned_allow_base or settings.finetuned_lora_path or settings.ip_adapter_enabled)

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

    def _load_finetuned_pipeline(self):
        if self._finetuned_pipeline is not None:
            return self._finetuned_pipeline

        import torch
        from diffusers import StableDiffusionImg2ImgPipeline

        if not self.finetuned_configured():
            raise RuntimeError(
                "Fine-tuned studio requires VISGEN_FINETUNED_LORA_PATH, "
                "VISGEN_IP_ADAPTER_ENABLED=true, or VISGEN_FINETUNED_ALLOW_BASE=true."
            )

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

        local_model = PROJECT_ROOT / "models" / settings.finetuned_model_id.replace("/", "__")
        model_source = str(local_model) if local_model.exists() else settings.finetuned_model_id
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_source, **kwargs)
        pipe = pipe.to(settings.device)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

        if settings.finetuned_lora_path:
            lora_kwargs = {}
            if settings.finetuned_lora_weight_name:
                lora_kwargs["weight_name"] = settings.finetuned_lora_weight_name
            pipe.load_lora_weights(settings.finetuned_lora_path, **lora_kwargs)

        if settings.ip_adapter_enabled:
            pipe.load_ip_adapter(
                settings.ip_adapter_repo,
                subfolder=settings.ip_adapter_subfolder,
                weight_name=settings.ip_adapter_weight_name,
            )
            pipe.set_ip_adapter_scale(settings.ip_adapter_scale)

        self._finetuned_pipeline = pipe
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
        styles: list[VariationStyle] | None,
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
        active_styles = styles or self.styles
        for idx, style in enumerate(active_styles, start=1):
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
        styles: list[VariationStyle] | None,
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
        active_styles = styles or self.styles
        with httpx.Client(timeout=180) as client:
            for idx, style in enumerate(active_styles, start=1):
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

    def _generate_modelslab_variations(
        self,
        input_path: Path,
        output_dir: Path,
        base_seed: int,
        preserve_faces: bool,
        styles: list[VariationStyle] | None,
    ) -> list[dict[str, Any]]:
        import base64
        import os

        import httpx

        api_key = settings.modelslab_api_key or os.getenv("MODELSLAB_API_KEY", "")
        if not api_key:
            raise RuntimeError("MODELSLAB_API_KEY or VISGEN_MODELSLAB_API_KEY is required for API studio mode.")

        model_id = (settings.modelslab_model_id or "").strip()
        if model_id == "gemini-3.1-t2i":
            return self._generate_modelslab_text_to_image_variations(
                input_path,
                output_dir,
                base_seed,
                preserve_faces,
                styles,
                api_key,
            )

        prepared = self._prepare_image(input_path)
        api_input = output_dir / "api_input.png"
        prepared.save(api_input)
        image_b64 = base64.b64encode(api_input.read_bytes()).decode("utf-8")
        active_styles = styles or self.styles
        width, height = self._parse_image_size(settings.openai_image_size)
        negative_prompt = (
            "blurry, low quality, distorted face, changed identity, waxy skin, malformed eyes, "
            "deformed teeth, extra facial features"
        )

        results: list[dict[str, Any]] = []
        with httpx.Client(timeout=240) as client:
            for idx, style in enumerate(active_styles, start=1):
                seed = base_seed + style.seed_offset
                payload = {
                    "key": api_key,
                    "prompt": self._api_edit_prompt(style, idx),
                    "negative_prompt": negative_prompt,
                    "init_image": image_b64,
                    "width": width,
                    "height": height,
                    "samples": "1",
                    "num_inference_steps": "30",
                    "safety_checker": "no",
                    "enhance_prompt": "yes",
                    "guidance_scale": max(4.0, float(style.guidance_scale)),
                    "strength": max(0.2, min(0.75, float(style.strength))),
                    "seed": str(seed),
                    "scheduler": settings.modelslab_scheduler,
                    "model_id": model_id,
                }
                response = client.post(f"{settings.modelslab_base_url.rstrip('/')}/img2img", json=payload)
                if response.status_code >= 400:
                    raise RuntimeError(f"Modelslab img2img failed: {response.status_code} {response.text}")

                payload_json = response.json()
                outputs = payload_json.get("output") or payload_json.get("images") or []
                if not outputs:
                    raise RuntimeError(f"Modelslab img2img returned no output: {payload_json}")

                image_ref = outputs[0]
                out_path = output_dir / f"variation_{idx}.png"
                if isinstance(image_ref, str) and image_ref.startswith("http"):
                    img_response = client.get(image_ref)
                    if img_response.status_code >= 400:
                        raise RuntimeError(
                            f"Modelslab output download failed: {img_response.status_code} {img_response.text}"
                        )
                    out_path.write_bytes(img_response.content)
                else:
                    image_data = base64.b64decode(str(image_ref))
                    out_path.write_bytes(image_data)

                face_count = len(self._detect_faces(prepared)) if preserve_faces else 0
                results.append(self._metadata(idx, out_path, seed, style, preserve_faces, face_count, "modelslab-api"))
        return results

    def _generate_modelslab_text_to_image_variations(
        self,
        input_path: Path,
        output_dir: Path,
        base_seed: int,
        preserve_faces: bool,
        styles: list[VariationStyle] | None,
        api_key: str,
    ) -> list[dict[str, Any]]:
        import base64

        import httpx

        prepared = self._prepare_image(input_path)
        active_styles = styles or self.styles
        results: list[dict[str, Any]] = []

        with httpx.Client(timeout=240) as client:
            for idx, style in enumerate(active_styles, start=1):
                seed = base_seed + style.seed_offset
                payload = {
                    "key": api_key,
                    "model_id": "gemini-3.1-t2i",
                    "prompt": self._api_text_to_image_prompt(style, idx),
                    "aspect_ratio": settings.modelslab_aspect_ratio,
                    "resolution": settings.modelslab_resolution,
                }
                response = client.post(settings.modelslab_t2i_url, json=payload)
                if response.status_code >= 400:
                    raise RuntimeError(f"Modelslab text-to-image failed: {response.status_code} {response.text}")

                payload_json = response.json()
                outputs = payload_json.get("output") or payload_json.get("images") or []
                if not outputs:
                    raise RuntimeError(f"Modelslab text-to-image returned no output: {payload_json}")

                image_ref = outputs[0]
                out_path = output_dir / f"variation_{idx}.png"
                if isinstance(image_ref, str) and image_ref.startswith("http"):
                    img_response = client.get(image_ref)
                    if img_response.status_code >= 400:
                        raise RuntimeError(
                            f"Modelslab output download failed: {img_response.status_code} {img_response.text}"
                        )
                    out_path.write_bytes(img_response.content)
                else:
                    image_data = base64.b64decode(str(image_ref))
                    out_path.write_bytes(image_data)

                generated = Image.open(out_path).convert("RGB")
                face_count = 0
                if preserve_faces:
                    generated, face_count = self._preserve_faces(prepared, generated)
                generated = self._studio_finish(generated)
                generated.save(out_path)

                results.append(
                    self._metadata(
                        idx,
                        out_path,
                        seed,
                        style,
                        preserve_faces,
                        face_count,
                        "modelslab-api-t2i",
                    )
                )

        return results

    def _parse_image_size(self, raw: str) -> tuple[int, int]:
        text = (raw or "512x512").lower().strip()
        if "x" not in text:
            return 512, 512
        left, right = text.split("x", 1)
        try:
            width = int(left)
            height = int(right)
            return width, height
        except ValueError:
            return 512, 512

    def _generate_finetuned_variations(
        self,
        input_path: Path,
        output_dir: Path,
        base_seed: int,
        preserve_faces: bool,
        styles: list[VariationStyle] | None,
    ) -> list[dict[str, Any]]:
        import torch

        pipe = self._load_finetuned_pipeline()
        image = self._prepare_image(input_path)
        negative_prompt = (
            "low quality, blurry, distorted face, changed identity, deformed eyes, "
            "asymmetrical face, malformed mouth, plastic skin, waxy skin, artifacts, "
            "watermark, text, extra facial features, bad teeth, oversaturated"
        )

        results: list[dict[str, Any]] = []
        active_styles = styles or self.styles
        for idx, style in enumerate(active_styles, start=1):
            seed = base_seed + style.seed_offset
            generator = torch.Generator(device=settings.device).manual_seed(seed)
            call_kwargs: dict[str, Any] = {
                "prompt": self._finetuned_prompt(style, idx),
                "negative_prompt": negative_prompt,
                "image": image,
                "strength": max(style.strength, 0.42),
                "guidance_scale": max(style.guidance_scale, 7.0),
                "num_inference_steps": settings.finetuned_steps,
                "generator": generator,
            }
            if settings.ip_adapter_enabled:
                call_kwargs["ip_adapter_image"] = image
            if settings.finetuned_lora_path:
                call_kwargs["cross_attention_kwargs"] = {"scale": settings.finetuned_lora_scale}

            result = pipe(**call_kwargs)
            generated = result.images[0].convert("RGB")
            face_count = 0
            if preserve_faces:
                generated, face_count = self._preserve_faces(image, generated)
            generated = self._studio_finish(generated)

            out_path = output_dir / f"variation_{idx}.png"
            generated.save(out_path)
            results.append(self._metadata(idx, out_path, seed, style, preserve_faces, face_count, "fine-tuned-local"))
        return results

    def _generate_demo_variations(
        self,
        input_path: Path,
        output_dir: Path,
        base_seed: int,
        preserve_faces: bool,
        styles: list[VariationStyle] | None,
    ) -> list[dict[str, Any]]:
        image = self._prepare_image(input_path)
        active_styles = styles or self.styles
        transforms = [
            lambda im: self._natural_lighting(im),
            lambda im: self._cinematic_tint(im),
            lambda im: self._studio_headshot(im),
            lambda im: self._editorial_polish(im),
            lambda im: self._soft_luxury_retouch(im),
        ]
        results: list[dict[str, Any]] = []
        for idx, (style, transform) in enumerate(zip(active_styles, transforms), start=1):
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

    def _api_text_to_image_prompt(self, style: VariationStyle, idx: int) -> str:
        return (
            f"Professional portrait photo, studio look {idx}: {style.label}. {style.prompt}. "
            "Preserve realistic face proportions, detailed eyes, natural skin texture, clean lighting, "
            "and production-ready portrait quality. Avoid blur, waxy skin, distorted anatomy, artifacts, and text."
        )

    def _finetuned_prompt(self, style: VariationStyle, idx: int) -> str:
        return (
            f"portrait_studio_lora look {idx}, {style.prompt}, professional photography, "
            "recognizable same person, natural skin texture, sharp eyes, realistic face geometry, "
            "cinematic but natural lighting, production-quality retouching"
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
        source_boxes = self._detect_faces(source)
        if not source_boxes:
            return generated, 0

        generated_boxes = self._detect_faces(generated)
        result = generated.copy()
        for source_box in source_boxes:
            target_box = self._best_matching_box(source_box, generated_boxes) or source_box
            source_crop_box = self._expanded_box(source.size, source_box, pad_x=0.22, pad_y=0.18)
            target_crop_box = self._expanded_box(generated.size, target_box, pad_x=0.22, pad_y=0.18)
            left, top, right, bottom = target_crop_box
            if right <= left or bottom <= top:
                continue

            source_crop = source.crop(source_crop_box)
            source_crop = source_crop.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
            face_canvas = result.copy()
            face_canvas.paste(source_crop, (left, top))
            mask = self._face_mask(generated.size, target_box)
            face_layer = Image.composite(face_canvas, result, mask)
            result = Image.blend(result, face_layer, alpha=settings.face_lock_blend)
        return result, len(source_boxes)

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
        x, _y, w, _h = box
        left, top, right, bottom = self._expanded_box(size, box, pad_x=0.22, pad_y=0.18)

        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((left, top, right, bottom), radius=max(24, w // 5), fill=190)
        return mask.filter(ImageFilter.GaussianBlur(radius=max(12, w // 10)))

    def _expanded_box(
        self,
        size: tuple[int, int],
        box: tuple[int, int, int, int],
        pad_x: float,
        pad_y: float,
    ) -> tuple[int, int, int, int]:
        width, height = size
        x, y, w, h = box
        px = int(w * pad_x)
        py = int(h * pad_y)
        return (
            max(0, x - px),
            max(0, y - py),
            min(width, x + w + px),
            min(height, y + h + py),
        )

    def _best_matching_box(
        self,
        source_box: tuple[int, int, int, int],
        candidates: list[tuple[int, int, int, int]],
    ) -> tuple[int, int, int, int] | None:
        if not candidates:
            return None
        sx, sy, sw, sh = source_box
        source_center = (sx + sw / 2, sy + sh / 2)

        def distance(candidate: tuple[int, int, int, int]) -> float:
            x, y, w, h = candidate
            center = (x + w / 2, y + h / 2)
            return (center[0] - source_center[0]) ** 2 + (center[1] - source_center[1]) ** 2

        return min(candidates, key=distance)

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
