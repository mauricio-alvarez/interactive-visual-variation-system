from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FFHQ_PATH_MARKERS = {"images1024x1024", "in-the-wild-images", "realign1024x1024"}


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def image_paths(raw_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            continue
        paths.extend(
            path
            for path in raw_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return sorted(paths)


def stable_id(path: Path) -> str:
    return hashlib.sha1(str(path).replace("\\", "/").encode("utf-8")).hexdigest()[:12]


def infer_identity(path: Path, raw_dirs: list[Path]) -> str:
    for raw_dir in raw_dirs:
        try:
            rel = path.relative_to(raw_dir)
        except ValueError:
            continue
        if FFHQ_PATH_MARKERS.intersection(rel.parts):
            return clean_id(f"ffhq_{path.stem}")
        if len(rel.parts) > 1:
            return clean_id(rel.parts[0])
    return clean_id(path.stem)


def clean_id(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in cleaned.split("_") if part) or "unknown"


def split_for_identity(identity_id: str, validation_ratio: float, test_ratio: float) -> str:
    bucket = int(hashlib.sha1(identity_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + validation_ratio:
        return "validation"
    return "train"


def detect_faces(image: Image.Image) -> list[tuple[int, int, int, int]]:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        return []
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(72, 72))
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def blur_variance(image: Image.Image) -> float:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def expanded_square(
    size: tuple[int, int],
    box: tuple[int, int, int, int],
    padding: float,
) -> tuple[int, int, int, int]:
    width, height = size
    x, y, w, h = box
    center_x = x + w / 2
    center_y = y + h / 2
    side = max(w, h) * (1 + padding * 2)
    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    right = int(round(center_x + side / 2))
    bottom = int(round(center_y + side / 2))

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height

    return max(0, left), max(0, top), min(width, right), min(height, bottom)


def process_image(
    path: Path,
    output_dir: Path,
    output_size: int,
    face_padding: float,
    min_source_size: int,
    min_face_ratio: float,
    max_face_ratio: float,
    blur_min: float,
) -> tuple[Path | None, dict[str, Any]]:
    try:
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception as exc:
        return None, {"reject_reason": f"unreadable: {exc}"}

    width, height = image.size
    if min(width, height) < min_source_size:
        return None, {"reject_reason": "source_too_small", "width": width, "height": height}

    faces = detect_faces(image)
    if not faces:
        return None, {"reject_reason": "no_face_detected", "width": width, "height": height}

    face = max(faces, key=lambda item: item[2] * item[3])
    face_ratio = max(face[2], face[3]) / min(width, height)
    if face_ratio < min_face_ratio:
        return None, {"reject_reason": "face_too_small", "face_ratio": face_ratio}
    if face_ratio > max_face_ratio:
        return None, {"reject_reason": "face_too_large", "face_ratio": face_ratio}

    blur = blur_variance(image)
    if blur < blur_min:
        return None, {"reject_reason": "too_blurry", "blur_variance": blur}

    crop_box = expanded_square(image.size, face, face_padding)
    cropped = image.crop(crop_box)
    cropped = ImageOps.fit(cropped, (output_size, output_size), method=Image.Resampling.LANCZOS)

    output_path = output_dir / f"{stable_id(path)}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output_path, quality=94, subsampling=1)
    return output_path, {
        "width": width,
        "height": height,
        "face_count": len(faces),
        "face_ratio": round(face_ratio, 4),
        "blur_variance": round(blur, 3),
        "crop_box": crop_box,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare consented portrait data for LoRA training.")
    parser.add_argument("--config", default="config/finetuning.yaml")
    parser.add_argument("--profile", default="sd15_local", choices=["sd15_local", "sdxl_production"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    dataset_cfg = config["dataset"]
    profile = config["profiles"][args.profile]
    raw_dirs = [ROOT / item for item in dataset_cfg["raw_dirs"]]
    processed_dir = ROOT / dataset_cfg["processed_dir"]
    metadata_dir = ROOT / dataset_cfg["metadata_dir"]
    metadata_dir.mkdir(parents=True, exist_ok=True)

    output_size = int(profile["resolution"])
    paths = image_paths(raw_dirs)
    if args.limit:
        paths = paths[: args.limit]

    default_caption = dataset_cfg["default_caption"]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for path in paths:
        identity_id = infer_identity(path, raw_dirs)
        split = split_for_identity(
            identity_id,
            float(dataset_cfg["validation_identity_ratio"]),
            float(dataset_cfg["test_identity_ratio"]),
        )
        split_dir = processed_dir / args.profile / split
        output_path, details = process_image(
            path=path,
            output_dir=split_dir,
            output_size=output_size,
            face_padding=float(dataset_cfg["face_padding"]),
            min_source_size=int(dataset_cfg["min_source_size"]),
            min_face_ratio=float(dataset_cfg["min_face_ratio"]),
            max_face_ratio=float(dataset_cfg["max_face_ratio"]),
            blur_min=float(dataset_cfg["blur_variance_min"]),
        )

        row = {
            "image_id": stable_id(path),
            "source_path": str(path.relative_to(ROOT)),
            "identity_id": identity_id,
            "split": split,
            "caption": default_caption,
            **details,
        }
        if output_path is None:
            rejected.append(row)
            continue

        row["path"] = str(output_path.relative_to(ROOT))
        accepted.append(row)

    write_csv(metadata_dir / f"{args.profile}_images.csv", accepted)
    write_csv(metadata_dir / f"{args.profile}_rejected.csv", rejected)
    write_metadata_jsonl(processed_dir / args.profile, accepted)

    summary = {
        "profile": args.profile,
        "raw_images": len(paths),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "splits": {
            split: sum(1 for item in accepted if item["split"] == split)
            for split in ["train", "validation", "test"]
        },
    }
    (metadata_dir / f"{args.profile}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata_jsonl(processed_root: Path, rows: list[dict[str, Any]]) -> None:
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        by_split[row["split"]].append(row)

    for split, split_rows in by_split.items():
        split_dir = processed_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = split_dir / "metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                file_name = Path(row["path"]).name
                handle.write(json.dumps({"file_name": file_name, "text": row["caption"]}) + "\n")


if __name__ == "__main__":
    main()
