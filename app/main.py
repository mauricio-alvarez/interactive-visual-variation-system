from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.settings import PROJECT_ROOT, settings
from app.schemas import DecisionRequest, DecisionResponse, GenerationResponse
from app.services.explainer import DecisionExplanationAgent, accepted_rejected_ids
from app.services.generator import ImageVariationGenerator
from app.services.storage import SessionStore, public_output_url

app = FastAPI(title=settings.project_name)
templates = Jinja2Templates(directory=PROJECT_ROOT / "app" / "templates")
store = SessionStore()
generator = ImageVariationGenerator()
explainer = DecisionExplanationAgent()

app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "app" / "static"), name="static")
app.mount("/outputs", StaticFiles(directory=PROJECT_ROOT / "outputs"), name="outputs")
app.mount("/examples", StaticFiles(directory=PROJECT_ROOT / "examples"), name="examples")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "project_name": settings.project_name,
            "default_mode": "diffusion",
        },
    )


@app.get("/api/health")
def health():
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else ""
    except Exception:
        cuda_available = False
        gpu_name = ""
    return {
        "status": "ok",
        "model_id": settings.model_id,
        "device": settings.device,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }


@app.post("/api/generate", response_model=GenerationResponse)
async def generate(
    image: UploadFile = File(...),
    mode: str = Form("diffusion"),
    base_seed: int = Form(4200),
    preserve_faces: bool = Form(True),
):
    if mode not in {"diffusion", "demo"}:
        raise HTTPException(status_code=400, detail="mode must be diffusion or demo")

    session_id, session_dir = store.create_session()
    input_path = await store.save_upload(image, session_dir)
    variations_dir = session_dir / "variations"

    try:
        variations = generator.generate(
            input_path,
            variations_dir,
            base_seed=base_seed,
            mode=mode,
            preserve_faces=preserve_faces,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response_variations = []
    for item in variations:
        image_path = Path(item["image_path"])
        response_variations.append({**item, "image_url": public_output_url(image_path)})

    record = {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "preserve_faces": preserve_faces,
        "input_image": str(input_path),
        "input_image_url": public_output_url(input_path),
        "variations": response_variations,
        "decisions": [],
        "summary": "",
    }
    store.write_record(session_dir, record)

    return {
        "session_id": session_id,
        "mode": mode,
        "preserve_faces": preserve_faces,
        "input_image_url": record["input_image_url"],
        "variations": response_variations,
    }


@app.post("/api/sessions/{session_id}/decisions", response_model=DecisionResponse)
def save_decisions(session_id: str, request: DecisionRequest):
    ids = sorted(item.variation_id for item in request.decisions)
    if ids != [1, 2, 3, 4, 5]:
        raise HTTPException(status_code=400, detail="Decisions must include variation IDs 1 through 5.")

    record = store.read_record(session_id)
    decisions = [item.model_dump() for item in request.decisions]
    record["decisions"] = decisions
    record["summary"] = explainer.explain(record)
    record["decided_at"] = datetime.now(timezone.utc).isoformat()

    record_path = store.write_record(settings.output_dir / session_id, record)
    accepted_ids, rejected_ids = accepted_rejected_ids(decisions)

    return {
        "session_id": session_id,
        "summary": record["summary"],
        "accepted_ids": accepted_ids,
        "rejected_ids": rejected_ids,
        "record_path": str(record_path),
    }
