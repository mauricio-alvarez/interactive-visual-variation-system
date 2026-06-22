# Evaluation

## Required criteria

The rubric gives points for system design, generative model, human-AI interaction, explainability, evaluation, ethics, and documentation.

## Metrics

Visual quality:

- Human rating from 1 to 5.
- Artifact count or qualitative defect notes.

Diversity:

- Pairwise inverse SSIM between the five generated images.
- Optional CLIP embedding distance if time permits.

Input fidelity:

- Human rating from 1 to 5.
- SSIM against the input for domains where structure should remain stable.

Decision consistency:

- Accepted/rejected IDs in the summary must match the JSON record.
- Score: exact match over all sessions.

Usability:

- Time to complete one session.
- Number of missed decisions.
- Short user questionnaire.

## Suggested user study

Use 3 to 5 participants. Each participant completes one session:

1. Upload or select an input image.
2. Review five outputs.
3. Accept or reject each variation.
4. Read the summary.
5. Rate image quality, diversity, interface clarity, and explanation trust.

