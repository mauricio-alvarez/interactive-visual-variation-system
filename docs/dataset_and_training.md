# Dataset And Training Plan

## Dataset choice

Recommended dataset: consented personal portraits or a licensed portrait subset.

Reason:

- It matches the professional studio goal.
- It supports direct evaluation of identity preservation, eyes, skin texture, lighting, and retouching.
- It can be ethically valid if the team uses consented images or a dataset with compatible license terms.
- It is feasible for small LoRA experiments when the subset is controlled.

Fallback dataset:

- AFHQ v2 or a small product/object dataset if the team decides to avoid human-subject images.
- Use only images with explicit permission or compatible public licenses.

## Preparation

1. Keep a small raw subset in `data/raw` and do not commit it.
2. Resize and center-crop to 512x512.
3. Remove corrupt, duplicate, or very low-quality images.
4. Save processed samples in `data/processed` for local experiments.
5. Document dataset source, license, filters, and known limitations.

## Advanced model path

Primary route:

- Keep `runwayml/stable-diffusion-v1-5` only as the local proof-of-training baseline.
- Use SDXL LoRA for the production fine-tuned studio path when 24 GB or larger GPU training is available.
- Add a face-reference conditioning layer for identity preservation instead of relying on style fine-tuning alone.
- Keep LoRA rank small for local experiments, for example 4 or 8, to fit an 8 GB GPU.

Recommended generation defaults:

- Resolution: 512x512.
- Precision: fp16.
- Steps: 28.
- Guidance scale: 6.7 to 7.2.
- Strength: 0.26 to 0.34 for portraits.
- Exactly five studio looks with fixed seed offsets.

## What to report

- Why the dataset fits the use case.
- What attributes are intended to vary.
- What attributes should be preserved, especially facial identity and eye/mouth geometry.
- Training or adaptation settings.
- Failure cases: over-editing, low diversity, artifacts, identity drift, or prompt mismatch.

See `docs/model_finetuning_plan.md` for the full fine-tuning dataset, training, and evaluation plan.
