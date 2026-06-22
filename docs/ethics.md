# Ethics

## Main risks

- Identity manipulation if human faces are used.
- Dataset bias in visual attributes, style, lighting, or subject representation.
- Misleading synthetic images if outputs are presented as real.
- User overtrust in generated explanations.
- Copyright or license problems when using scraped images.

## Mitigations

- Prefer consented portraits, licensed portrait datasets, AFHQ, or consented object/product images for the main demo.
- Avoid private or sensitive images.
- Keep generated images clearly labeled as AI-generated variations.
- Store user decisions transparently in JSON.
- Make the explanation agent quote the decision record rather than invent rationale.
- Document dataset source, license, and limitations.
- Re-enable the model safety checker before accepting unknown public uploads or human-subject images.

## Report angle

The strongest ethical discussion is specific:

- What dataset was used.
- Why that dataset reduces or introduces risk.
- What the model fails to preserve.
- What should not be inferred from the generated images.
- How the UI keeps the human in control of final selection.
