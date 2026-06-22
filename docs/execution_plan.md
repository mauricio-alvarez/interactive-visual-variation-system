# Execution Plan

## Phase 1 - Scope

- Domain: consented personal portraits or licensed portrait examples.
- Goal: generate five professional photographer-style edits and support human selection.
- Output: architecture diagram, constraints, and success criteria.

## Phase 2 - Data

- Select dataset.
- Clean and preprocess images.
- Record source, license, and filters.
- Produce before/after examples.

## Phase 3 - Generation

- Verify CUDA with `scripts/check_gpu.py`.
- Run the FastAPI app.
- Test `Fast demo` mode for UI.
- Download model weights or let first generation download them.
- Test `GPU studio` mode.
- Optionally fine-tune LoRA and set `VISGEN_LORA_PATH`.

## Phase 4 - Human Feedback

- Upload an input image.
- Generate exactly five studio looks.
- Keep face lock enabled for portraits.
- Keep or pass every look.
- Store decisions in `outputs/sessions/<session_id>/record.json`.

## Phase 5 - Explanation

- Generate a final text summary from the decision record.
- Verify accepted and rejected IDs match the JSON exactly.
- Include screenshots and the JSON record in the report.

## Phase 6 - Evaluation

- Measure visual diversity with inverse SSIM.
- Evaluate face correctness: identity, eyes, mouth, skin texture, and artifacts.
- Add human ratings for quality, usefulness, and fidelity.
- Check explanation consistency.
- Discuss limitations and ethical risks.

## Phase 7 - Delivery

- Code repository.
- Technical report, maximum five pages.
- Architecture diagram.
- Example input and five outputs.
- Demo video or live demo.
