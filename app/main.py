from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.settings import PROJECT_ROOT, settings
from app.schemas import DecisionRequest, DecisionResponse, GenerationResponse, RefinementResponse
from app.services.explainer import DecisionExplanationAgent, accepted_rejected_ids
from app.services.generator import ImageVariationGenerator
from app.services.preferences import PreferenceProfiler
from app.services.storage import SessionStore, public_output_url

app = FastAPI(title=settings.project_name)
templates = Jinja2Templates(directory=PROJECT_ROOT / "app" / "templates")
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
frontend_assets = frontend_dist / "assets"
store = SessionStore()
generator = ImageVariationGenerator()
explainer = DecisionExplanationAgent()
preference_profiler = PreferenceProfiler()

app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "app" / "static"), name="static")
app.mount("/outputs", StaticFiles(directory=PROJECT_ROOT / "outputs"), name="outputs")
app.mount("/examples", StaticFiles(directory=PROJECT_ROOT / "examples"), name="examples")
if frontend_assets.exists():
    app.mount("/assets", StaticFiles(directory=frontend_assets), name="frontend-assets")


def _api_key_configured() -> bool:
    return generator.api_key_configured()


def _finetuned_configured() -> bool:
    return generator.finetuned_configured()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    frontend_index = frontend_dist / "index.html"
    if frontend_index.exists():
        return HTMLResponse(frontend_index.read_text(encoding="utf-8"))

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
    provider = generator.api_provider()
    llm_provider = (settings.explainer_provider or "rule").strip().lower() or "rule"
    if provider == "modelslab":
        api_model = settings.modelslab_model_id
    elif provider == "huggingface":
        api_model = settings.huggingface_model_id
    elif provider == "modal":
        api_model = (
            settings.modal_endpoint_url
            or os.getenv("VISGEN_MODAL_ENDPOINT_URL", "")
            or os.getenv("MODAL_ENDPOINT_URL", "")
            or "modal-endpoint"
        )
    else:
        api_model = settings.openai_image_model

    if llm_provider == "groq":
        llm_model = settings.groq_model
        llm_key_configured = bool(settings.groq_api_key or os.getenv("GROQ_API_KEY"))
    else:
        llm_model = "rules-v1"
        llm_key_configured = True

    return {
        "status": "ok",
        "model_id": settings.model_id,
        "device": settings.device,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "api_provider": provider,
        "api_key_configured": _api_key_configured(),
        "api_image_model": api_model,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_key_configured": llm_key_configured,
        "finetuned_ready": _finetuned_configured(),
        "finetuned_model_id": settings.finetuned_model_id,
        "identity_adapter_enabled": settings.ip_adapter_enabled,
    }


@app.post("/api/generate", response_model=GenerationResponse)
async def generate(
    image: UploadFile = File(...),
    mode: str = Form("diffusion"),
    base_seed: int = Form(4200),
    preserve_faces: bool = Form(True),
):
    if mode not in {"diffusion", "demo", "api", "finetuned"}:
        raise HTTPException(status_code=400, detail="mode must be diffusion, demo, api, or finetuned")
    if mode == "api" and not _api_key_configured():
        provider = generator.api_provider()
        detail = (
            "API studio requires MODELSLAB_API_KEY or VISGEN_MODELSLAB_API_KEY."
            if provider == "modelslab"
            else (
                "API studio requires HF_TOKEN, HUGGINGFACEHUB_API_TOKEN, or VISGEN_HUGGINGFACE_API_KEY."
                if provider == "huggingface"
                else (
                    "API studio requires VISGEN_MODAL_ENDPOINT_URL or MODAL_ENDPOINT_URL."
                    if provider == "modal"
                    else "API studio requires OPENAI_API_KEY or VISGEN_OPENAI_API_KEY."
                )
            )
        )
        raise HTTPException(
            status_code=400,
            detail=detail,
        )
    if mode == "finetuned" and not _finetuned_configured():
        raise HTTPException(
            status_code=400,
            detail=(
                "Fine-tuned studio requires VISGEN_FINETUNED_LORA_PATH, "
                "VISGEN_IP_ADAPTER_ENABLED=true, or VISGEN_FINETUNED_ALLOW_BASE=true."
            ),
        )

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
        "base_seed": base_seed,
        "preserve_faces": preserve_faces,
        "input_image": str(input_path),
        "input_image_url": public_output_url(input_path),
        "variations": response_variations,
        "decisions": [],
        "summary": "",
        "generation_history": [
            {
                "round": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "base_seed": base_seed,
                "preserve_faces": preserve_faces,
                "input_image": str(input_path),
                "input_image_url": public_output_url(input_path),
                "variations": response_variations,
            }
        ],
        "decision_history": [],
        "refinement_history": [],
        "summary_history": [],
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
    explanation = explainer.explain_with_metadata(record)
    record["summary"] = explanation["summary"]
    record["explanation_provider"] = explanation["provider"]
    record["explanation_model"] = explanation["model"]
    record["decided_at"] = datetime.now(timezone.utc).isoformat()

    record.setdefault("decision_history", []).append(
        {
            "round": int(record.get("refinement_round", 0)),
            "decided_at": record["decided_at"],
            "decisions": decisions,
        }
    )
    record.setdefault("summary_history", []).append(
        {
            "timestamp": record["decided_at"],
            "kind": "decision",
            "summary": record["summary"],
            "provider": record.get("explanation_provider", "rule"),
            "model": record.get("explanation_model", "rules-v1"),
        }
    )

    record_path = store.write_record(settings.output_dir / session_id, record)
    accepted_ids, rejected_ids = accepted_rejected_ids(decisions)

    return {
        "session_id": session_id,
        "summary": record["summary"],
        "explanation_provider": record.get("explanation_provider", "rule"),
        "explanation_model": record.get("explanation_model", "rules-v1"),
        "accepted_ids": accepted_ids,
        "rejected_ids": rejected_ids,
        "record_path": str(record_path),
    }


@app.post("/api/sessions/{session_id}/refine", response_model=RefinementResponse)
def refine_session(session_id: str):
    record = store.read_record(session_id)
    decisions = record.get("decisions", [])
    accepted_ids, _ = accepted_rejected_ids(decisions)
    if not accepted_ids:
        raise HTTPException(status_code=400, detail="Refinement requires at least one kept look.")

    mode = str(record.get("mode") or "diffusion")
    preserve_faces = bool(record.get("preserve_faces", True))
    input_path = Path(record["input_image"])
    round_index = int(record.get("refinement_round", 0)) + 1
    initial_seed = int(record.get("base_seed", 4200))
    base_seed = initial_seed + (round_index * 100)
    variations_dir = settings.output_dir / session_id / f"refinements/round_{round_index:02d}"
    variations_dir.mkdir(parents=True, exist_ok=True)

    try:
        preference_profile = preference_profiler.build(decisions, record.get("variations", []))
        refined_styles = generator.refined_styles_from_feedback_ai(
            decisions,
            record.get("variations", []),
            preference_profile=preference_profile,
        )
        variations = generator.generate(
            input_path,
            variations_dir,
            base_seed=base_seed,
            mode=mode,
            preserve_faces=preserve_faces,
            styles=refined_styles,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response_variations = []
    for item in variations:
        image_path = Path(item["image_path"])
        response_variations.append({**item, "image_url": public_output_url(image_path)})

    try:
        refinement_explanation = explainer.explain_refinement_with_metadata(
            record,
            accepted_ids,
            preference_profile,
            round_index,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    refinement_summary = refinement_explanation["summary"]

    record["variations"] = response_variations
    record["summary"] = refinement_summary
    record["explanation_provider"] = refinement_explanation["provider"]
    record["explanation_model"] = refinement_explanation["model"]
    record["refinement_round"] = round_index
    record["preference_profile"] = preference_profile
    record["refined_at"] = datetime.now(timezone.utc).isoformat()
    record.setdefault("generation_history", []).append(
        {
            "round": round_index,
            "created_at": record["refined_at"],
            "mode": mode,
            "base_seed": base_seed,
            "preserve_faces": preserve_faces,
            "input_image": str(input_path),
            "input_image_url": record.get("input_image_url", ""),
            "variations": response_variations,
            "source": "refinement",
        }
    )
    record.setdefault("refinement_history", []).append(
        {
            "round": round_index,
            "refined_at": record["refined_at"],
            "accepted_ids": accepted_ids,
            "base_seed": base_seed,
            "summary": refinement_summary,
            "preference_profile": preference_profile,
        }
    )
    record.setdefault("summary_history", []).append(
        {
            "timestamp": record["refined_at"],
            "kind": "refinement",
            "summary": refinement_summary,
            "provider": record.get("explanation_provider", "unknown"),
            "model": record.get("explanation_model", "unknown"),
        }
    )
    store.write_record(settings.output_dir / session_id, record)

    return {
        "session_id": session_id,
        "summary": refinement_summary,
        "mode": mode,
        "preserve_faces": preserve_faces,
        "base_seed": base_seed,
        "variations": response_variations,
    }
