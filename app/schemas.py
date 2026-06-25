from pydantic import BaseModel, Field


class VariationInfo(BaseModel):
    id: int
    image_url: str
    image_path: str
    seed: int
    label: str
    provider: str = "local"
    prompt: str
    strength: float
    guidance_scale: float
    face_preservation: bool = True
    detected_faces: int = 0


class GenerationResponse(BaseModel):
    session_id: str
    mode: str
    preserve_faces: bool
    input_image_url: str
    variations: list[VariationInfo]


class DecisionInput(BaseModel):
    variation_id: int = Field(ge=1, le=5)
    decision: str = Field(pattern="^(accepted|rejected)$")
    reason: str = ""


class DecisionRequest(BaseModel):
    decisions: list[DecisionInput] = Field(min_length=5, max_length=5)


class DecisionResponse(BaseModel):
    session_id: str
    summary: str
    accepted_ids: list[int]
    rejected_ids: list[int]
    record_path: str


class RefinementResponse(BaseModel):
    session_id: str
    summary: str
    mode: str
    preserve_faces: bool
    base_seed: int
    variations: list[VariationInfo]
