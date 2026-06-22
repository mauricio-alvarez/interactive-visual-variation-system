# Deployment

## Current Deployment Target

This app is deployable as a FastAPI web service. The default container is UI-first and preview/API-mode friendly, so it can run on ordinary web hosts without downloading model weights or installing the full diffusion stack. GPU studio mode still requires a CUDA host, model weights, and the GPU PyTorch dependencies.

## Local Production Run

```powershell
.\.venv\Scripts\python.exe scripts\serve.py
```

Set a custom port:

```powershell
$env:PORT="8001"
.\.venv\Scripts\python.exe scripts\serve.py
```

## Docker Preview Build

```powershell
docker build -t ai-portrait-studio .
docker run --rm -p 8000:8000 ai-portrait-studio
```

Open `http://127.0.0.1:8000`.

## API Studio Deployment

Use API studio when local model quality or variation diversity is not enough. Configure one of these keys outside the repository:

```text
OPENAI_API_KEY=sk-...
VISGEN_OPENAI_API_KEY=sk-...
```

Optional provider settings:

```text
VISGEN_OPENAI_IMAGE_MODEL=gpt-image-2
VISGEN_OPENAI_IMAGE_SIZE=1024x1024
VISGEN_OPENAI_IMAGE_QUALITY=medium
```

The app posts each uploaded portrait to the image edit endpoint and stores the returned image in the session output folder. Do not commit `.env`, uploads, generated outputs, or API keys.

## GPU Deployment Notes

For GPU production, use a CUDA-capable host and install the CUDA PyTorch wheel before `requirements.txt`:

```powershell
pip install -r requirements-gpu-cu128.txt
pip install -r requirements.txt
```

Then place or download model weights under `models/` and set:

```text
VISGEN_DEVICE=cuda
```

## Production Checklist

- Use HTTPS behind a reverse proxy or managed platform.
- Store API keys only in environment secrets.
- Keep uploaded and generated images outside the repository.
- Put `outputs/sessions` on persistent storage if sessions must survive restarts.
- Re-enable or replace the safety checker before accepting public uploads.
- Add authentication before exposing the editor to the open internet.
- Set upload size limits at the proxy/platform layer.
- Use consented portraits or licensed datasets only.
