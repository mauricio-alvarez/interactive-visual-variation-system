from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from urllib.parse import parse_qs, urlencode, urlparse
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_ffhq_downloader(repo_dir: Path) -> ModuleType:
    script_path = repo_dir / "download_ffhq.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"Missing {script_path}. Clone https://github.com/NVlabs/ffhq-dataset.git first."
        )

    spec = importlib.util.spec_from_file_location("ffhq_downloader", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ffhq_downloader"] = module
    spec.loader.exec_module(module)
    return module


def confirmed_drive_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc != "drive.google.com" or parsed.path != "/uc":
        return url

    file_ids = parse_qs(parsed.query).get("id")
    if not file_ids:
        return url
    return "https://drive.usercontent.google.com/download?" + urlencode(
        {"id": file_ids[0], "confirm": "t"}
    )


def rebase_file_spec(file_spec: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    rebased = dict(file_spec)
    rebased["file_url"] = confirmed_drive_url(file_spec["file_url"])
    rebased["file_path"] = str(output_dir / file_spec["file_path"])
    return rebased


def selected_items(
    json_data: dict[str, Any],
    count: int,
    category: str,
    seed: int,
    shuffle: bool,
) -> list[tuple[str, dict[str, Any]]]:
    items = [
        (key, item)
        for key, item in json_data.items()
        if category == "all" or item.get("category") == category
    ]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    return items[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a bounded FFHQ 1024x1024 subset.")
    parser.add_argument(
        "--repo-dir",
        default="data/portrait_finetune/raw/licensed/ffhq-dataset",
        help="Local clone of NVlabs/ffhq-dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/portrait_finetune/raw/licensed/ffhq",
        help="Where FFHQ metadata and selected images should be stored.",
    )
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--category", choices=["training", "validation", "all"], default="training")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()

    repo_dir = (ROOT / args.repo_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    downloader = load_ffhq_downloader(repo_dir)
    metadata_spec = rebase_file_spec(downloader.json_spec, output_dir)
    license_spec = rebase_file_spec(downloader.license_specs["json"], output_dir)

    print("Downloading FFHQ metadata if needed...")
    downloader.download_files([metadata_spec, license_spec], num_threads=args.num_threads)

    metadata_path = output_dir / downloader.json_spec["file_path"]
    print(f"Parsing metadata: {metadata_path}")
    with metadata_path.open("rb") as handle:
        json_data = json.load(handle)

    chosen = selected_items(
        json_data=json_data,
        count=args.count,
        category=args.category,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    image_specs = [rebase_file_spec(item["image"], output_dir) for _, item in chosen]
    image_specs.append(rebase_file_spec(downloader.license_specs["images"], output_dir))
    total_bytes = sum(spec["file_size"] for spec in image_specs)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.relative_to(ROOT)),
                "category": args.category,
                "requested_images": args.count,
                "selected_images": len(chosen),
                "estimated_download_gb": round(total_bytes / 1024**3, 3),
                "shuffle": args.shuffle,
                "seed": args.seed,
            },
            indent=2,
        )
    )

    downloader.download_files(image_specs, num_threads=args.num_threads)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
