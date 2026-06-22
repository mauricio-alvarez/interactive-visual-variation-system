# Dataset And Training Plan

## Dataset choice

Recommended dataset: AFHQ v2 subset.

Reason:

- It supports face-like image variation analysis without manipulating real human identity.
- It has clear visual attributes: pose, texture, lighting, background, and color.
- It reduces privacy and consent risk compared with human face datasets.
- It is feasible for small LoRA experiments.

Fallback dataset:

- A small product/object dataset collected by the team.
- Use only images with explicit permission or compatible public licenses.

## Preparation

1. Keep a small raw subset in `data/raw` and do not commit it.
2. Resize and center-crop to 512x512.
3. Remove corrupt, duplicate, or very low-quality images.
4. Save processed samples in `data/processed` for local experiments.
5. Document dataset source, license, filters, and known limitations.

## Advanced model path

Primary route:

- Start from `runwayml/stable-diffusion-v1-5`.
- Use img2img for input-conditioned generation.
- Add LoRA fine-tuning on the selected dataset if time permits.
- Keep LoRA rank small, for example 4 or 8, to fit an 8 GB GPU.

Recommended generation defaults:

- Resolution: 512x512.
- Precision: fp16.
- Steps: 28.
- Guidance scale: 7.0 to 8.0.
- Strength: 0.42 to 0.58.
- Exactly five variations with fixed seed offsets.

## What to report

- Why the dataset fits the use case.
- What attributes are intended to vary.
- What attributes should be preserved.
- Training or adaptation settings.
- Failure cases: over-editing, low diversity, artifacts, identity drift, or prompt mismatch.

