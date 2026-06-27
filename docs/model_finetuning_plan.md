# Model Fine-Tuning Plan

## Problem

The current local GPU mode uses Stable Diffusion 1.5 img2img with conservative strengths. That keeps the image close to the input, but it also causes two product failures:

- The five outputs are visually too similar.
- Faces drift, distort, or no longer look like the uploaded person when the model changes the image enough to create real variety.

Fine-tuning alone will not fully solve identity preservation from a single uploaded image. The model plan must separate two goals:

- Studio style quality: lighting, color grade, backgrounds, retouching, and photographic polish.
- Identity preservation: face shape, eyes, mouth, age, and recognizable subject consistency.

## Recommendation

Use a two-layer model strategy:

1. Train a portrait studio LoRA for professional photography style and output diversity.
2. Add reference-face conditioning for identity preservation during inference.

The LoRA should teach the model what a high-quality studio result looks like. It should not memorize people. Identity should come from the uploaded image through an adapter or face-reference pathway.

## Base Model Choice

### Preferred production path

- Base: SDXL-family portrait model with compatible license.
- Training: SDXL LoRA on a 24 GB or larger GPU.
- Deployment: local inference with the LoRA if performance is acceptable, otherwise API studio remains the production fallback.

Reason: SDXL has better native face and portrait quality than SD 1.5, especially at 1024px.

### Local-only fallback

- Base: current SD 1.5-compatible model.
- Training: SD 1.5 LoRA at 512px on the RTX 3060 Ti 8 GB.
- Purpose: prove the data and evaluation loop before paying for stronger compute.

Reason: this is feasible locally, but it is unlikely to fully reach the professional portrait quality target.

## Dataset Strategy

### Production dataset

Use a consented or explicitly licensed portrait dataset. This is the only recommended production-grade source.

Minimum target:

- 2,000 to 5,000 professional portrait images for style LoRA.
- 100 to 300 identities for identity evaluation, with 6 to 12 images per identity.
- 30 to 50 held-out identities that never appear in training.
- 5 fixed validation prompts per held-out identity.
- 1024px source quality when training SDXL; 512px processed copies for SD 1.5 fallback.

Required variety:

- Frontal, three-quarter, and mild side poses.
- Natural window light, softbox, cinematic rim light, editorial contrast, and beauty lighting.
- Neutral, warm, dark, textured, and studio-gray backgrounds.
- Different skin tones, ages, hair styles, eyewear, and face shapes.
- Single clear subject per image.

Reject:

- Unlicensed web scrapes.
- Celebrity datasets for production training.
- Minors unless consent and legal basis are explicit.
- Watermarked images.
- Images with heavy filters, face apps, plastic skin, or extreme makeup unless explicitly needed.
- Duplicates or near-duplicates.

### Research-only references

These can help benchmark or prototype, but should not become the production training source:

- FFHQ: useful for non-commercial research and face-quality benchmarking; it contains 70,000 1024px face images and license metadata, but the dataset license is non-commercial/share-alike and it is not intended for facial-recognition development.
- CelebA: useful for attributes and identity split concepts, but the official agreement is non-commercial research only and prohibits commercial exploitation or redistribution.
- Open Images: useful for broad Creative Commons imagery and possible background/person filtering, but it is not a curated professional portrait dataset.

## Dataset Folder Layout

```text
data/portrait_finetune/
  raw/
    licensed/
    consented/
  metadata/
    images.csv
    identities.csv
    licenses.csv
    consent_records.md
  processed/
    train/
    validation/
    test/
  identity_eval/
    subject_0001/
    subject_0002/
models/
  lora/
    portrait_sdxl/
reports/
  finetune/
```

Do not commit raw data, processed portraits, identity folders, consent records, or trained weights unless the license explicitly permits it.

## Metadata Schema

`images.csv`:

```text
image_id,path,identity_id,split,license_id,consent_id,width,height,pose,lighting,background,crop,caption
```

`identities.csv`:

```text
identity_id,split,image_count,consent_scope,notes
```

`licenses.csv`:

```text
license_id,source,license_name,license_url,commercial_allowed,attribution_required,redistribution_allowed,notes
```

Do not store real names unless legally required by the consent workflow. Use anonymous IDs.

## Captioning Rules

Captions should describe photographic conditions, not identity.

Good:

```text
professional portrait photo, medium close-up, natural window lighting, soft neutral background, realistic skin texture, sharp eyes, 85mm lens
```

Bad:

```text
photo of Jane Doe, attractive woman, celebrity style, perfect face
```

Caption fields to include:

- Shot type: headshot, medium close-up, editorial portrait, beauty portrait.
- Lighting: window light, softbox, rim light, clamshell, low key.
- Background: seamless gray, warm neutral, dark textured, outdoor blurred.
- Camera feel: 85mm lens, shallow depth of field, crisp eyes.
- Retouch level: natural skin texture, subtle retouching.

## Training Phases

### Phase 0: Baseline audit

Run the existing app on 20 consented test portraits:

- Generate five outputs per portrait.
- Save decisions and failure notes.
- Measure diversity and face similarity.
- Keep these outputs as the pre-fine-tuning baseline.

### Phase 1: Dataset build

Tasks:

- Collect or license the portrait set.
- Verify consent and license metadata.
- Remove duplicates and low-quality images.
- Detect and crop faces.
- Generate captions.
- Split by identity, not by image.

Acceptance gate:

- No identity appears in more than one split.
- At least 95% of images pass face detection.
- At least 90% of captions correctly describe lighting/background/crop.

### Phase 2: Local proof LoRA

Purpose: prove the pipeline before cloud spend.

Suggested settings:

- Base model: SD 1.5-compatible local model.
- Resolution: 512.
- LoRA rank: 8.
- Batch size: 1.
- Gradient accumulation: 4.
- Mixed precision: fp16.
- Training steps: 3,000 to 6,000.
- Learning rate: start at `1e-4`, reduce if overfitting appears.
- Validation each epoch for the current Diffusers text-to-image LoRA script.

Expected result:

- Better lighting and style separation.
- Still limited face fidelity.

### Phase 3: SDXL portrait LoRA

Purpose: production-quality studio style.

Suggested settings:

- Base model: SDXL-compatible portrait base with compatible license.
- Resolution: 1024.
- LoRA rank: 8 or 16.
- Batch size: 1.
- Gradient accumulation: 4 to 8.
- Mixed precision: fp16 or bf16.
- Training steps: 5,000 to 12,000.
- Use gradient checkpointing.
- Use validation prompts for all five studio looks.

Compute:

- Recommended: 24 GB or larger GPU.
- Local RTX 3060 Ti 8 GB is not the recommended training target for SDXL.

### Phase 4: Identity preservation layer

Add a reference-face pathway rather than trying to memorize every possible user.

Options:

- IP-Adapter FaceID or similar face-embedding adapter.
- InstantID-style identity conditioning.
- ControlNet pose/depth to keep head geometry stable.
- Face restoration only as a final repair pass, not as the main identity solution.

Acceptance gate:

- The uploaded face remains recognizable in at least 4 of 5 outputs.
- Face detector succeeds on at least 95% of outputs.
- Human review marks identity as same person for at least 85% of outputs.

### Phase 5: App integration

Add a new generation mode:

```text
Fine-tuned studio
```

Runtime stack:

- Load base model.
- Load portrait LoRA.
- Apply face-reference conditioning.
- Generate five studio looks.
- Run face-quality checks.
- Fall back to API studio when local confidence is low.

Implemented app configuration:

- `VISGEN_FINETUNED_MODEL_ID`
- `VISGEN_FINETUNED_LORA_PATH`
- `VISGEN_FINETUNED_LORA_WEIGHT_NAME`
- `VISGEN_FINETUNED_LORA_SCALE`
- `VISGEN_IP_ADAPTER_ENABLED`
- `VISGEN_IP_ADAPTER_REPO`
- `VISGEN_IP_ADAPTER_SUBFOLDER`
- `VISGEN_IP_ADAPTER_WEIGHT_NAME`
- `VISGEN_IP_ADAPTER_SCALE`

## Evaluation Metrics

Diversity:

- Pairwise SSIM across the five outputs.
- CLIP image embedding distance.
- Human rating: "Do these five looks feel different?"

Identity:

- Face embedding similarity between input and output.
- Face detector success rate.
- Landmark plausibility.
- Human same-person rating.

Quality:

- Artifact count around eyes, teeth, mouth, and hairline.
- Skin texture naturalness.
- Prompt/look match.
- Preference ranking against current GPU studio and API studio.

Production gates:

- At least 4 of 5 outputs are recognizable as the input subject.
- At least 3 of 5 outputs are meaningfully different.
- No severe eye/mouth deformation in accepted outputs.
- API studio remains better on no more than 30% of validation cases before switching the local model to default.

## Inputs Required Before Training

Required:

- Final license target: research demo only or production/commercial use.
- Dataset source or budget for a licensed/consented portrait set.
- Consent scope for human subjects.
- Compute decision: local SD 1.5 proof only, or SDXL LoRA on 24 GB+ GPU.
- Quality thresholds for identity, diversity, and artifact tolerance.

Recommended default if no further input is provided:

- Build a consented 2,000-image portrait style set.
- Build a held-out 50-subject identity validation set.
- Train local SD 1.5 LoRA as a pipeline proof.
- Train SDXL LoRA on cloud GPU for production.
- Integrate face-reference conditioning before making fine-tuned local mode the default.

## Sources Checked

- FFHQ official dataset page: 70,000 images, 1024px, non-commercial/share-alike dataset license, per-image license metadata, and privacy removal process.
- CelebA official dataset page: more than 200,000 celebrity face images, 10,177 identities, non-commercial research-only agreement.
- Open Images V7 official page: large-scale annotated image dataset; useful for filtering but not a portrait-specialized training set.
- Hugging Face Diffusers DreamBooth and LoRA docs: LoRA is lightweight, DreamBooth is sensitive to hyperparameters, prior preservation helps diversity, and text-encoder training for faces needs substantially more VRAM.
