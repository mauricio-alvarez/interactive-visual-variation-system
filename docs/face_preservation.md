# Face Preservation

## Problem

Img2img can improve lighting and color, but portrait edits can also drift identity or damage face details. Common failures are changed eye shape, asymmetry, plastic skin, malformed mouth, or a face that no longer looks like the source person.

## Current Strategy

The app uses conservative portrait generation settings:

- Lower img2img strengths, from 0.26 to 0.34.
- Prompts that explicitly request same face, same identity, realistic skin texture, and sharp eyes.
- Negative prompts for distorted faces, changed identity, deformed eyes, malformed mouth, and plastic skin.
- Optional face lock enabled by default.

Face lock uses OpenCV frontal-face detection on the prepared input image. For each detected face, the app creates a feathered mask and softly blends the original facial region back into the generated output. This is not a full face restoration model, but it is practical and local.

## Why This Helps

The diffusion model can still improve the surrounding image, lighting, color, and background while the most identity-sensitive region remains anchored to the source. This is useful for profile photos and headshots.

## Limitations

- Haar-cascade detection works best with frontal faces.
- It may miss side profiles, occluded faces, or very small faces.
- If the diffusion model moves the head too much, same-coordinate blending can look imperfect.
- Very low-quality source faces still need better restoration, not only preservation.

## Future Upgrade

The next technical upgrade should be a stronger portrait stack:

- Face detection with RetinaFace or MediaPipe.
- Identity guidance with IP-Adapter FaceID or InstantID.
- Face restoration with GFPGAN or CodeFormer.
- ControlNet pose/depth for preserving geometry.
