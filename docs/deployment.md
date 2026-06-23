# Deployment

## Current Deployment Target

This app is deployable as a FastAPI web service. The default container is UI-first and preview/API-mode friendly, so it can run on ordinary web hosts without downloading model weights or installing the full diffusion stack. GPU studio mode still requires a CUDA host, model weights, and the GPU PyTorch dependencies.

## Local Production Run

Build the React interface first:

```powershell
cd frontend
npm install
npm run build
cd ..
```

```powershell
.\.venv\Scripts\python.exe scripts\serve.py
```

Set a custom port:

```powershell
$env:PORT="8001"
.\.venv\Scripts\python.exe scripts\serve.py
```

## Docker Preview Build

The Dockerfile builds the React frontend in a Node stage and copies the compiled assets into the FastAPI image.

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

## Fine-Tuned Studio Deployment

Fine-tuned studio is disabled until weights or an identity adapter are configured:

```text
VISGEN_FINETUNED_MODEL_ID=runwayml/stable-diffusion-v1-5
VISGEN_FINETUNED_LORA_PATH=models/lora/portrait_sd15_local
VISGEN_FINETUNED_LORA_WEIGHT_NAME=pytorch_lora_weights.safetensors
VISGEN_IP_ADAPTER_ENABLED=true
```

Use `VISGEN_FINETUNED_ALLOW_BASE=true` only for local debugging. It enables the mode without trained weights and should not be treated as production quality.

Training dependencies are intentionally separate from runtime dependencies:

```powershell
pip install -r requirements-training.txt
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
