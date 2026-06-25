from __future__ import annotations

import base64
import binascii
import io
import math
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
import json

import modal
from pydantic import BaseModel, Field


APP_NAME = "image-generation-service-v1"
MODEL_ID = "stabilityai/sdxl-turbo"
GPU_TYPE = "A100-80GB"
HF_CACHE_PATH = "/root/.cache/huggingface"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("modal_qwen_inference")


def _safe_preview(text: str | None, limit: int = 180) -> str:
    if not text:
        return ""
    sanitized = " ".join(text.split())
    return sanitized[:limit]


def _log(level: int, event: str, **fields: Any) -> None:
    logger.log(level, "%s | %s", event, json.dumps(fields, default=str))


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "torch==2.5.1",
        "diffusers==0.31.0",
        "transformers==4.47.1",
        "huggingface_hub==0.26.5",
        "Pillow==11.0.0",
    )
)

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("hf-cache-image-gen", create_if_missing=True)


class InferenceRequest(BaseModel):
    request_id: str | None = None
    prompt: str = Field(min_length=1)
    input_image_base64: str | None = None
    negative_prompt: str | None = None
    width: int = Field(default=1024, ge=512, le=1536)
    height: int = Field(default=1024, ge=512, le=1536)
    strength: float = Field(default=0.45, ge=0.05, le=0.95)
    num_inference_steps: int = Field(default=4, ge=1, le=8)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=10.0)
    seed: int | None = None


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=900,
    scaledown_window=300,
    volumes={HF_CACHE_PATH: cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class ImageGenerationService:
    @modal.enter()
    def load(self) -> None:
        start = time.perf_counter()
        _log(
            logging.INFO,
            "startup_loading_model",
            model=MODEL_ID,
            gpu_type=GPU_TYPE,
            hf_cache=HF_CACHE_PATH,
        )

        _log(
            logging.INFO,
            "startup_environment",
            python_version=os.sys.version,
            cwd=os.getcwd(),
            pid=os.getpid(),
        )

        try:
            import diffusers
            import transformers
            import torch

            _log(
                logging.INFO,
                "startup_package_versions",
                diffusers=diffusers.__version__,
                transformers=transformers.__version__,
                torch=torch.__version__,
            )
        except Exception:
            logger.exception("[startup] Failed to introspect package versions")

        import torch
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        try:
            self.pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                variant="fp16",
                cache_dir=HF_CACHE_PATH,
            )
            self.pipeline_t2i = self.pipeline_t2i.to("cuda")
            self.pipeline_t2i.enable_attention_slicing()

            # Build an img2img pipeline from the same loaded components.
            self.pipeline_i2i = AutoPipelineForImage2Image.from_pipe(self.pipeline_t2i)
            self.pipeline_i2i.enable_attention_slicing()
        except Exception:
            logger.error("[startup] Pipeline initialization failed")
            logger.error(traceback.format_exc())
            raise

        _log(
            logging.INFO,
            "startup_model_loaded",
            startup_seconds=round(time.perf_counter() - start, 3),
        )

    @modal.method()
    def generate(self, request: InferenceRequest) -> dict[str, Any]:
        import torch
        from PIL import Image as PILImage

        request_id = request.request_id or str(uuid4())
        begin = time.perf_counter()
        _log(
            logging.INFO,
            "generate_request_received",
            request_id=request_id,
            prompt_chars=len(request.prompt),
            has_input_image=bool(request.input_image_base64),
            width=request.width,
            height=request.height,
            strength=request.strength,
            steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            has_seed=request.seed is not None,
        )

        _log(
            logging.INFO,
            "generate_prompt_received",
            request_id=request_id,
            user_preview=_safe_preview(request.prompt),
            negative_preview=_safe_preview(request.negative_prompt),
        )

        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(request.seed)

        mode = "text2img"
        init_image = None
        effective_steps = request.num_inference_steps
        if request.input_image_base64:
            mode = "img2img"
            raw_b64 = request.input_image_base64
            if raw_b64.startswith("data:") and "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(raw_b64, validate=False)
                init_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
                init_image = init_image.resize((request.width, request.height), PILImage.LANCZOS)
            except (binascii.Error, OSError, ValueError) as exc:
                raise ValueError("Invalid input_image_base64. Expected a valid base64 image.") from exc

            # Diffusers img2img can produce empty timesteps when int(steps * strength) == 0.
            # Auto-bump steps to guarantee at least one denoising step.
            if int(effective_steps * request.strength) < 1:
                effective_steps = max(effective_steps, math.ceil(1.0 / request.strength))
                _log(
                    logging.INFO,
                    "generate_steps_autobumped",
                    request_id=request_id,
                    requested_steps=request.num_inference_steps,
                    effective_steps=effective_steps,
                    strength=request.strength,
                )

        try:
            if mode == "img2img":
                result = self.pipeline_i2i(
                    prompt=request.prompt,
                    image=init_image,
                    negative_prompt=request.negative_prompt,
                    strength=request.strength,
                    num_inference_steps=effective_steps,
                    guidance_scale=request.guidance_scale,
                    generator=generator,
                )
            else:
                result = self.pipeline_t2i(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    width=request.width,
                    height=request.height,
                    num_inference_steps=effective_steps,
                    guidance_scale=request.guidance_scale,
                    generator=generator,
                )
        except Exception:
            _log(logging.ERROR, "generate_failed", request_id=request_id)
            logger.error(traceback.format_exc())
            raise

        image = result.images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        elapsed = round(time.perf_counter() - begin, 3)

        _log(
            logging.INFO,
            "generate_succeeded",
            request_id=request_id,
            elapsed_seconds=elapsed,
            mode=mode,
            image_bytes=buffer.tell(),
            output_width=image.width,
            output_height=image.height,
        )

        return {
            "model": MODEL_ID,
            "image_base64": image_b64,
            "mime_type": "image/png",
            "request_id": request_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "meta": {
                "mode": mode,
                "width": image.width,
                "height": image.height,
                "strength": request.strength,
                "steps": effective_steps,
                "requested_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
            },
        }


model = ImageGenerationService()


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def infer(request: InferenceRequest) -> dict[str, Any]:
    from fastapi import HTTPException

    request_id = str(uuid4())
    _log(
        logging.INFO,
        "infer_incoming_request",
        request_id=request_id,
        prompt_chars=len(request.prompt),
        has_input_image=bool(request.input_image_base64),
        width=request.width,
        height=request.height,
        strength=request.strength,
        steps=request.num_inference_steps,
    )

    try:
        request_with_id = request.model_copy(update={"request_id": request_id})
        result = model.generate.remote(request_with_id)
        _log(logging.INFO, "infer_request_completed", request_id=request_id)
        return result
    except ValueError as exc:
        _log(logging.WARNING, "infer_request_rejected", request_id=request_id, detail=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log(
            logging.ERROR,
            "infer_internal_error",
            request_id=request_id,
            error_type=type(exc).__name__,
        )
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error. request_id={request_id}",
        ) from exc


# Deploy with:
#   modal deploy modal_qwen_inference.py
# Run once locally for testing:
#   modal run modal_qwen_inference.py::infer --prompt "cinematic portrait photo of a woman with dramatic lighting"
