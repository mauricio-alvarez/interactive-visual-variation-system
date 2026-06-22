# AI Portrait Studio

FastAPI project for the advanced version of UTEC Project 2. The app works as a local AI portrait studio: it takes a user image, creates exactly five professional photographer-style looks, collects keep/pass feedback, and produces a faithful decision summary.

## Current design

- Local GPU-first generation with Stable Diffusion img2img through `diffusers`.
- RTX 3060 Ti friendly defaults: 512x512, fp16, 28 steps, attention slicing.
- Five studio presets from `config/generation.yaml`: natural lighting, cinematic tint, studio headshot, editorial polish, and soft luxury retouch.
- Face lock: optional OpenCV face detection plus soft facial-region blending to reduce identity drift and malformed faces.
- Browser UI for upload, generation, feedback, and summary.
- JSON session records under `outputs/sessions`.
- Tracked visual example under `examples/session_001`.

## Setup

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\check_gpu.py
```

The venv has already been created in this workspace. To recreate it:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8001`.

Use `GPU studio` for the real local GPU path. Use `Fast demo` only for UI testing and documentation screenshots.

The local default disables the built-in Stable Diffusion safety checker because it can false-positive on the non-human classroom demo images. Re-enable it with `VISGEN_DISABLE_SAFETY_CHECKER=false` before accepting unknown, public, or human-subject uploads.

## Model weights

The first CUDA generation downloads the model from Hugging Face unless it already exists in the local cache. If Hugging Face requires acceptance or authentication for the selected model, authenticate with:

```powershell
huggingface-cli login
```

## Project docs

- `docs/architecture.md`
- `docs/dataset_and_training.md`
- `docs/deployment.md`
- `docs/execution_plan.md`
- `docs/evaluation.md`
- `docs/ethics.md`
- `docs/face_preservation.md`

## Deployment

The deployable preview container runs the UI and Preview mode without requiring model weights or the full diffusion stack:

```powershell
docker build -t ai-portrait-studio .
docker run --rm -p 8000:8000 ai-portrait-studio
```

GPU studio deployment needs a CUDA host, the GPU PyTorch wheel, and local model weights. See `docs/deployment.md`.
