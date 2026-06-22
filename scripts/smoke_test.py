import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app


def main() -> None:
    client = TestClient(app)
    image_path = Path("examples/session_001/input.png")

    with image_path.open("rb") as handle:
        response = client.post(
            "/api/generate",
            data={"mode": "demo", "base_seed": "9000"},
            files={"image": ("input.png", handle, "image/png")},
        )
    print(f"generate status: {response.status_code}")
    response.raise_for_status()
    payload = response.json()
    print(f"session: {payload['session_id']}")
    print(f"variations: {len(payload['variations'])}")

    decisions = {
        "decisions": [
            {"variation_id": 1, "decision": "accepted", "reason": "clear"},
            {"variation_id": 2, "decision": "rejected", "reason": "too colorful"},
            {"variation_id": 3, "decision": "accepted", "reason": "balanced"},
            {"variation_id": 4, "decision": "accepted", "reason": "sharp"},
            {"variation_id": 5, "decision": "rejected", "reason": "dark"},
        ]
    }
    decision_response = client.post(f"/api/sessions/{payload['session_id']}/decisions", json=decisions)
    print(f"decision status: {decision_response.status_code}")
    decision_response.raise_for_status()
    print(decision_response.json()["summary"])


if __name__ == "__main__":
    main()
