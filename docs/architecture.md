# Architecture

## Goal

The system receives one user image, produces exactly five professional studio-style portrait edits, collects keep/pass feedback for every look, and generates a final explanation tied to the recorded decisions.

## Components

```mermaid
flowchart LR
    A["Image upload"] --> B["Preprocess 512x512 RGB"]
    B --> C{"Generation mode"}
    C --> D["Preview transforms"]
    C --> E["Local Stable Diffusion img2img"]
    C --> F["OpenAI image edit API"]
    D --> G["Five studio looks"]
    E --> G
    F --> G
    G --> H["Face lock metadata"]
    H --> I["FastAPI browser UI"]
    I --> J["Keep/pass decisions"]
    J --> K["Session record JSON"]
    K --> L["Explanation agent"]
    L --> M["Final summary"]
```

## Runtime modules

- `app/main.py`: FastAPI endpoints and web UI wiring.
- `app/services/generator.py`: provider switch, CUDA diffusion generator, OpenAI image-edit provider, demo-mode studio transforms, and face-preservation pass.
- `app/services/storage.py`: session folders and JSON records.
- `app/services/explainer.py`: faithful explanation agent based on stored decisions.
- `app/services/evaluation.py`: starter diversity metric.
- `frontend/src`: React, TypeScript, Tailwind, and shadcn-style studio interface. FastAPI serves `frontend/dist` when it exists.
- `config/generation.yaml`: selected dataset, model, and five generation parameter sets.

## Session contract

Every session stores:

- Original uploaded image.
- Five generated professional studio looks.
- Seed, prompt, strength, and guidance scale for each variation.
- Provider used for each variation.
- Face-lock flag and detected face count.
- One accept/reject decision for each variation.
- Optional user reason per decision.
- Final summary.
