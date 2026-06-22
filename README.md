# Interactive Visual Variation System

FastAPI project for the advanced version of UTEC Project 2: generate exactly five visual variations from a user image, collect human accept/reject feedback, and produce a faithful decision summary.

## Current design

- Local GPU-first generation with Stable Diffusion img2img through `diffusers`.
- RTX 3060 Ti friendly defaults: 512x512, fp16, 28 steps, attention slicing.
- Five fixed variation styles from `config/generation.yaml`.
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

Open `http://127.0.0.1:8000`.

Use `CUDA diffusion` for the real local GPU path. Use `Fast demo` only for UI testing and documentation screenshots.

## Model weights

The first CUDA generation downloads the model from Hugging Face unless it already exists in the local cache. If Hugging Face requires acceptance or authentication for the selected model, authenticate with:

```powershell
huggingface-cli login
```

## Project docs

- `docs/architecture.md`
- `docs/dataset_and_training.md`
- `docs/execution_plan.md`
- `docs/evaluation.md`
- `docs/ethics.md`

