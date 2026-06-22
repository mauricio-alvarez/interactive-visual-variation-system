import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.explainer import DecisionExplanationAgent
from app.services.generator import ImageVariationGenerator


SESSION = ROOT / "examples" / "session_001"


def create_input(path: Path) -> None:
    image = Image.new("RGB", (512, 512), (236, 240, 243))
    draw = ImageDraw.Draw(image)
    draw.rectangle((76, 92, 436, 420), fill=(218, 226, 232), outline=(88, 101, 112), width=4)
    draw.ellipse((156, 132, 356, 332), fill=(176, 194, 207), outline=(63, 87, 107), width=5)
    draw.polygon([(256, 94), (306, 182), (206, 182)], fill=(18, 107, 95))
    draw.rectangle((188, 342, 324, 386), fill=(42, 52, 62))
    draw.text((128, 444), "Input image", fill=(32, 42, 52), font=ImageFont.load_default())
    image.save(path)


def main() -> None:
    SESSION.mkdir(parents=True, exist_ok=True)
    input_path = SESSION / "input.png"
    create_input(input_path)

    generator = ImageVariationGenerator()
    variations = generator.generate(input_path, SESSION, base_seed=4200, mode="demo")

    decisions = [
        {"variation_id": 1, "decision": "accepted", "reason": "Best balance between realism and fidelity."},
        {"variation_id": 2, "decision": "accepted", "reason": "Useful color style for presentation."},
        {"variation_id": 3, "decision": "rejected", "reason": "Too flat compared with the original."},
        {"variation_id": 4, "decision": "accepted", "reason": "Sharper details improve visual quality."},
        {"variation_id": 5, "decision": "rejected", "reason": "Tint changes the intended mood too much."},
    ]
    record = {
        "session_id": "session_001",
        "input_image": str(input_path),
        "variations": variations,
        "decisions": decisions,
    }
    summary = DecisionExplanationAgent().explain(record)
    (SESSION / "record.json").write_text(__import__("json").dumps(record | {"summary": summary}, indent=2), encoding="utf-8")
    (SESSION / "summary.txt").write_text(summary, encoding="utf-8")
    print(f"Created demo assets in {SESSION}")


if __name__ == "__main__":
    main()
