from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def accelerate_executable() -> str:
    executable_name = "accelerate.exe" if os.name == "nt" else "accelerate"
    candidate = Path(sys.executable).with_name(executable_name)
    if candidate.exists():
        return str(candidate)
    return "accelerate"


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_dataset(train_dir: Path) -> dict[str, Any]:
    metadata_path = train_dir / "metadata.jsonl"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory does not exist: {train_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run scripts/prepare_portrait_dataset.py first."
        )

    rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing = [row["file_name"] for row in rows if not (train_dir / row["file_name"]).exists()]
    if missing:
        raise FileNotFoundError(f"Metadata references missing files: {missing[:5]}")
    if len(rows) < 100:
        raise ValueError(
            f"Only {len(rows)} training images found. Use at least 100 for a smoke run and 2,000+ for useful style training."
        )
    return {"images": len(rows), "metadata": str(metadata_path)}


def script_path(examples_dir: Path, script_name: str) -> Path:
    candidates = [
        examples_dir / script_name,
        examples_dir / "text_to_image" / script_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find Diffusers training script. Clone diffusers and pass "
        "--diffusers-examples path\\to\\diffusers\\examples\\text_to_image."
    )


def build_command(
    profile: dict[str, Any],
    train_dir: Path,
    validation_prompt: str,
    training_script: Path,
    report_to: str,
) -> list[str]:
    command = [
        accelerate_executable(),
        "launch",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--mixed_precision",
        str(profile["mixed_precision"]),
        "--dynamo_backend",
        "no",
        str(training_script),
        "--pretrained_model_name_or_path",
        str(profile["base_model"]),
        "--train_data_dir",
        str(train_dir),
        "--resolution",
        str(profile["resolution"]),
        "--center_crop",
        "--random_flip",
        "--train_batch_size",
        str(profile["train_batch_size"]),
        "--gradient_accumulation_steps",
        str(profile["gradient_accumulation_steps"]),
        "--learning_rate",
        str(profile["learning_rate"]),
        "--lr_scheduler",
        "cosine",
        "--lr_warmup_steps",
        "0",
        "--max_train_steps",
        str(profile["max_train_steps"]),
        "--checkpointing_steps",
        str(profile["checkpointing_steps"]),
        "--validation_prompt",
        validation_prompt,
        "--validation_epochs",
        str(profile["validation_epochs"]),
        "--rank",
        str(profile["rank"]),
        "--mixed_precision",
        str(profile["mixed_precision"]),
        "--seed",
        str(profile["seed"]),
        "--output_dir",
        str(ROOT / profile["output_dir"]),
        "--report_to",
        report_to,
    ]
    if profile.get("gradient_checkpointing"):
        command.append("--gradient_checkpointing")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch portrait LoRA training with Diffusers examples.")
    parser.add_argument("--config", default="config/finetuning.yaml")
    parser.add_argument("--profile", default="sd15_local", choices=["sd15_local", "sdxl_production"])
    parser.add_argument("--diffusers-examples", default="")
    parser.add_argument("--validation-prompt-index", type=int, default=0)
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--run", action="store_true", help="Actually execute the command. Default is dry run.")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    profile = config["profiles"][args.profile]
    processed_root = ROOT / config["dataset"]["processed_dir"] / args.profile
    train_dir = processed_root / "train"
    dataset_summary = validate_dataset(train_dir)

    examples_dir = Path(args.diffusers_examples) if args.diffusers_examples else Path("external/diffusers/examples/text_to_image")
    if not examples_dir.is_absolute():
        examples_dir = ROOT / examples_dir
    training_script = script_path(examples_dir, profile["train_script"])

    prompts = config["validation_prompts"]
    prompt = prompts[args.validation_prompt_index % len(prompts)]
    command = build_command(profile, train_dir, prompt, training_script, args.report_to)

    print("Dataset:")
    print(json.dumps(dataset_summary, indent=2))
    print("\nCommand:")
    print(" ".join(shlex.quote(part) for part in command))

    if not args.run:
        print("\nDry run only. Add --run to start training.")
        return

    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
