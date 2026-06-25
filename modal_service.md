# Modal Service Documentation

## Overview

This project includes a Modal-hosted image generation service implemented in [modal_qwen_inference.py](modal_qwen_inference.py).

The service now supports two generation modes with the same endpoint:

1. Text-to-Image (T2I): prompt -> generated image
2. Image-to-Image (I2I): prompt + input image -> refined/generated image

Current deployed app name:
- image-generation-service-v1

Primary model:
- stabilityai/sdxl-turbo

## Runtime Stack

The Modal image installs these core packages:

- torch==2.5.1
- diffusers==0.31.0
- transformers==4.47.1
- huggingface_hub==0.26.5
- Pillow==11.0.0
- fastapi[standard]

Infrastructure settings:

- GPU: A100-80GB
- Cache volume: hf-cache-image-gen
- HF cache path: /root/.cache/huggingface
- Modal secret: huggingface-secret

## Service Architecture

The service uses one FastAPI endpoint exposed through Modal:

- infer (POST)

Request flow:

1. Request enters infer endpoint
2. Request ID is generated and logged
3. Request is forwarded to ImageGenerationService.generate
4. Generation mode is selected:
   - text2img when input_image_base64 is absent
   - img2img when input_image_base64 is present
5. Image output is encoded to base64 PNG
6. JSON response is returned

## API Reference

### Endpoint

POST https://juan-prochazka--image-generation-service-v1-infer.modal.run

### Request Fields

Required:

- prompt: string (non-empty)

Optional:

- input_image_base64: string (base64 image, optional data URL prefix supported)
- negative_prompt: string
- width: integer (default 1024, min 512, max 1536)
- height: integer (default 1024, min 512, max 1536)
- strength: float (default 0.45, min 0.05, max 0.95)
- num_inference_steps: integer (default 4, min 1, max 8)
- guidance_scale: float (default 0.0, min 0.0, max 10.0)
- seed: integer

### Response Fields

- model: string
- image_base64: string (PNG-encoded image content)
- mime_type: string (image/png)
- request_id: string
- timestamp_utc: ISO timestamp
- meta: object
  - mode: text2img or img2img
  - width: integer
  - height: integer
  - strength: float
  - steps: integer (effective runtime steps)
  - requested_steps: integer (requested by client)
  - guidance_scale: float
  - seed: integer or null

## Example Requests

### Text-to-Image

curl -sS -L -X POST "https://juan-prochazka--image-generation-service-v1-infer.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "editorial portrait, soft cinematic lighting, high detail",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 2,
    "guidance_scale": 0.0,
    "seed": 7
  }'

### Image-to-Image

curl -sS -L -X POST "https://juan-prochazka--image-generation-service-v1-infer.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "photoreal portrait refinement, preserve identity, subtle studio look",
    "input_image_base64": "<BASE64_IMAGE>",
    "width": 1024,
    "height": 1024,
    "strength": 0.45,
    "num_inference_steps": 2,
    "guidance_scale": 0.0,
    "seed": 7
  }'

## Logging and Observability

Structured log events include:

- startup_loading_model
- startup_environment
- startup_package_versions
- startup_model_loaded
- infer_incoming_request
- generate_request_received
- generate_prompt_received
- generate_steps_autobumped
- generate_succeeded
- generate_failed
- infer_request_completed
- infer_request_rejected
- infer_internal_error

Use request_id to correlate logs for one request across endpoint and generation steps.

## Important Stability Fix

Issue observed:

- img2img could fail when int(num_inference_steps * strength) == 0, causing a runtime tensor reshape error.

Fix implemented:

- The service auto-bumps effective steps for img2img to guarantee at least one denoising step.
- Example: requested_steps=2 with strength=0.45 becomes effective steps=3.

This is surfaced in response metadata:

- meta.steps (effective)
- meta.requested_steps (input)

## Deployment Workflow (Required)

Always run in this order:

1. Cleanup
2. Deploy
3. Test

### 1) Cleanup

Stop active apps before deploy:

python -m modal app list --json
python -m modal app stop -y <APP_ID>

### 2) Deploy

python -m modal deploy modal_qwen_inference.py

### 3) Test

Run both smoke tests:

- T2I request should return HTTP 200 and image_base64
- I2I request should return HTTP 200 and meta.mode=img2img

## Troubleshooting

If T2I works but I2I fails:

1. Check response body for request_id
2. Fetch logs for app and filter by request_id
3. Confirm input_image_base64 is valid image data
4. Verify deploy actually used the latest file
5. Ensure cleanup was performed before deploy

Useful commands:

python -m modal app list
python -m modal app logs image-generation-service-v1 --timestamps

## File Ownership

Main service source:

- [modal_qwen_inference.py](modal_qwen_inference.py)

This document:

- [modal_service.md](modal_service.md)
