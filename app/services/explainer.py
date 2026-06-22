from __future__ import annotations

from typing import Any


class DecisionExplanationAgent:
    """Faithful explanation layer for human feedback records."""

    def explain(self, record: dict[str, Any]) -> str:
        decisions = record.get("decisions", [])
        accepted = [d for d in decisions if d["decision"] == "accepted"]
        rejected = [d for d in decisions if d["decision"] == "rejected"]

        accepted_ids = ", ".join(str(d["variation_id"]) for d in accepted) or "none"
        rejected_ids = ", ".join(str(d["variation_id"]) for d in rejected) or "none"

        reasons = [
            f"Variation {d['variation_id']}: {d.get('reason', '').strip()}"
            for d in decisions
            if d.get("reason", "").strip()
        ]
        reason_text = " User notes: " + " | ".join(reasons) if reasons else ""

        return (
            f"The user accepted variation(s): {accepted_ids}. "
            f"The user rejected variation(s): {rejected_ids}. "
            "The final selection reflects the recorded accept/reject feedback and should be "
            "used as the decision trace for this session."
            f"{reason_text}"
        )


def accepted_rejected_ids(decisions: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    accepted = [item["variation_id"] for item in decisions if item["decision"] == "accepted"]
    rejected = [item["variation_id"] for item in decisions if item["decision"] == "rejected"]
    return accepted, rejected

