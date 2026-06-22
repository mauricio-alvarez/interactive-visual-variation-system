import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.generator import ImageVariationGenerator


def main() -> None:
    input_path = ROOT / "examples" / "session_001" / "input.png"
    output_dir = ROOT / "outputs" / "sessions" / "diffusion_smoke" / "variations"
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = ImageVariationGenerator()
    results = generator.generate(input_path, output_dir, base_seed=12000, mode="diffusion")
    print(f"generated: {len(results)}")
    for item in results:
        print(item["image_path"])


if __name__ == "__main__":
    main()

