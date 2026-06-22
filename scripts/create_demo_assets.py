import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.explainer import DecisionExplanationAgent
from app.services.generator import ImageVariationGenerator


SESSION = ROOT / "examples" / "session_001"


def create_input(path: Path) -> None:
    image = Image.new("RGB", (512, 512), (226, 232, 236))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 512, 512), fill=(224, 231, 235))
    draw.ellipse((58, 44, 454, 438), fill=(210, 220, 226))
    draw.rectangle((154, 346, 358, 512), fill=(45, 57, 68))
    draw.ellipse((142, 86, 370, 338), fill=(207, 174, 145), outline=(103, 82, 69), width=4)
    draw.pieslice((128, 56, 384, 266), 180, 360, fill=(42, 34, 32))
    draw.ellipse((190, 180, 220, 210), fill=(34, 42, 49))
    draw.ellipse((292, 180, 322, 210), fill=(34, 42, 49))
    draw.line((256, 202, 244, 258, 265, 258), fill=(121, 85, 70), width=4)
    draw.arc((218, 260, 294, 310), 8, 172, fill=(126, 56, 62), width=5)
    draw.text((154, 454), "Synthetic portrait input", fill=(32, 42, 52), font=ImageFont.load_default())
    image.save(path)


def main() -> None:
    SESSION.mkdir(parents=True, exist_ok=True)
    input_path = SESSION / "input.png"
    create_input(input_path)

    generator = ImageVariationGenerator()
    variations = generator.generate(input_path, SESSION, base_seed=4200, mode="demo")

    decisions = [
        {"variation_id": 1, "decision": "accepted", "reason": "Most natural portrait lighting."},
        {"variation_id": 2, "decision": "accepted", "reason": "Cinematic mood works for a profile image."},
        {"variation_id": 3, "decision": "accepted", "reason": "Cleanest studio headshot look."},
        {"variation_id": 4, "decision": "rejected", "reason": "Too much contrast for this face."},
        {"variation_id": 5, "decision": "accepted", "reason": "Soft retouch keeps the face approachable."},
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
