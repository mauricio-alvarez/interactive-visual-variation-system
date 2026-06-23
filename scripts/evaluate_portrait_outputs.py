from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_image(path: Path, size: int = 512) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)


def detect_faces(image: Image.Image) -> list[tuple[int, int, int, int]]:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        return []
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(72, 72))
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def largest_face_crop(image: Image.Image) -> Image.Image | None:
    faces = detect_faces(image)
    if not faces:
        return None
    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    pad_x = int(w * 0.25)
    pad_y = int(h * 0.25)
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image.width, x + w + pad_x)
    bottom = min(image.height, y + h + pad_y)
    return ImageOps.fit(image.crop((left, top, right, bottom)), (256, 256), method=Image.Resampling.LANCZOS)


def gray_array(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)


def ssim(a: Image.Image, b: Image.Image) -> float:
    return float(structural_similarity(gray_array(a), gray_array(b), data_range=255))


def evaluate_session(session_dir: Path, gates: dict[str, Any]) -> dict[str, Any]:
    record = json.loads((session_dir / "record.json").read_text(encoding="utf-8"))
    input_path = Path(record["input_image"])
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    source = load_image(input_path)
    source_face = largest_face_crop(source)

    rows: list[dict[str, Any]] = []
    output_images: list[Image.Image] = []
    face_scores: list[float] = []
    detected_faces = 0

    for item in record["variations"]:
        image_path = Path(item["image_path"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        output = load_image(image_path)
        output_images.append(output)
        output_face = largest_face_crop(output)
        has_face = output_face is not None
        detected_faces += int(has_face)
        face_similarity = ssim(source_face, output_face) if source_face is not None and output_face is not None else 0.0
        face_scores.append(face_similarity)
        rows.append(
            {
                "variation_id": item["id"],
                "label": item["label"],
                "provider": item.get("provider", ""),
                "full_image_ssim": round(ssim(source, output), 4),
                "face_similarity_ssim": round(face_similarity, 4),
                "face_detected": has_face,
            }
        )

    pair_scores = [ssim(a, b) for a, b in combinations(output_images, 2)]
    avg_pairwise_ssim = sum(pair_scores) / len(pair_scores) if pair_scores else 1.0
    diversity_score = 1.0 - avg_pairwise_ssim
    face_detection_rate = detected_faces / max(1, len(output_images))
    same_identity_rate = sum(1 for score in face_scores if score >= 0.34) / max(1, len(face_scores))
    distinct_looks = sum(1 for score in pair_scores if score < 0.82)

    report = {
        "session_id": record["session_id"],
        "mode": record["mode"],
        "provider_rows": rows,
        "metrics": {
            "diversity_score": round(diversity_score, 4),
            "avg_pairwise_ssim": round(avg_pairwise_ssim, 4),
            "distinct_pair_count": distinct_looks,
            "face_detection_rate": round(face_detection_rate, 4),
            "same_identity_rate_proxy": round(same_identity_rate, 4),
        },
        "gates": {
            "diversity_pass": distinct_looks >= int(gates["min_distinct_looks"]),
            "face_detection_pass": face_detection_rate >= float(gates["min_face_detection_rate"]),
            "identity_proxy_pass": same_identity_rate >= float(gates["min_same_identity_rate"]),
        },
    }
    report["gates"]["overall_pass"] = all(report["gates"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate portrait output diversity and face stability.")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--config", default="config/finetuning.yaml")
    parser.add_argument("--output-dir", default="reports/finetune")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    session_dir = Path(args.session_dir)
    if not session_dir.is_absolute():
        session_dir = ROOT / session_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = evaluate_session(session_dir, config["acceptance_gates"])
    report_path = output_dir / f"{report['session_id']}_evaluation.json"
    rows_path = output_dir / f"{report['session_id']}_variations.csv"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["provider_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(report["provider_rows"])

    print(json.dumps({"report": str(report_path), "metrics": report["metrics"], "gates": report["gates"]}, indent=2))


if __name__ == "__main__":
    main()
