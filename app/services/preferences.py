from __future__ import annotations

from typing import Any


class PreferenceProfiler:
    """Derive lightweight user preference signals from decisions and notes."""

    _TRAIT_KEYWORDS = {
        "natural lighting": ["natural", "daylight", "window", "realistic"],
        "cinematic mood": ["cinematic", "film", "dramatic", "moody"],
        "warm tones": ["warm", "golden", "sunset", "amber"],
        "cool tones": ["cool", "teal", "blue", "cold"],
        "clean background": ["clean", "minimal", "plain", "seamless"],
        "dark background": ["dark", "low-key", "shadow", "black"],
        "bright background": ["bright", "high-key", "airy", "light"],
        "soft skin texture": ["soft", "smooth", "retouch"],
        "sharp detail": ["sharp", "crisp", "detail", "texture"],
    }

    def build(self, decisions: list[dict[str, Any]], variations: list[dict[str, Any]]) -> dict[str, Any]:
        variation_map = {int(item.get("id", 0)): item for item in variations}
        liked_scores = {trait: 0.0 for trait in self._TRAIT_KEYWORDS}
        disliked_scores = {trait: 0.0 for trait in self._TRAIT_KEYWORDS}

        accepted_strengths: list[float] = []
        rejected_strengths: list[float] = []
        accepted_guidance: list[float] = []
        rejected_guidance: list[float] = []

        for item in decisions:
            var_id = int(item.get("variation_id", 0))
            variation = variation_map.get(var_id, {})
            decision = str(item.get("decision", "")).strip()
            reason_text = str(item.get("reason", "")).strip().lower()
            source_text = " ".join(
                [
                    str(variation.get("label", "")).lower(),
                    str(variation.get("prompt", "")).lower(),
                    reason_text,
                ]
            )

            if decision == "accepted":
                accepted_strengths.append(float(variation.get("strength", 0.4)))
                accepted_guidance.append(float(variation.get("guidance_scale", 7.0)))
                self._accumulate(liked_scores, source_text, weight=1.0)
            elif decision == "rejected":
                rejected_strengths.append(float(variation.get("strength", 0.4)))
                rejected_guidance.append(float(variation.get("guidance_scale", 7.0)))
                self._accumulate(disliked_scores, source_text, weight=1.0)

        liked_traits, avoided_traits = self._select_disjoint_traits(liked_scores, disliked_scores)

        strength_shift = self._clamp(
            self._mean(accepted_strengths) - self._mean(rejected_strengths),
            -0.08,
            0.08,
        )
        guidance_shift = self._clamp(
            self._mean(accepted_guidance) - self._mean(rejected_guidance),
            -1.0,
            1.0,
        )

        prompt_directive = self._build_directive(liked_traits, prefix="Emphasize")
        negative_directive = self._build_directive(avoided_traits, prefix="Avoid")

        return {
            "liked_traits": liked_traits,
            "avoided_traits": avoided_traits,
            "prompt_directive": prompt_directive,
            "negative_directive": negative_directive,
            "strength_shift": round(strength_shift, 3),
            "guidance_shift": round(guidance_shift, 3),
        }

    def _accumulate(self, target: dict[str, float], text: str, weight: float) -> None:
        if not text:
            return
        for trait, keywords in self._TRAIT_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                target[trait] += weight

    def _top_traits(self, scores: dict[str, float], limit: int = 3) -> list[str]:
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return [trait for trait, score in ranked if score > 0][:limit]

    def _select_disjoint_traits(
        self,
        liked_scores: dict[str, float],
        disliked_scores: dict[str, float],
        limit: int = 3,
    ) -> tuple[list[str], list[str]]:
        # Use a net score so a trait cannot simultaneously rank as strongly liked and disliked.
        net_scores = {
            trait: liked_scores.get(trait, 0.0) - disliked_scores.get(trait, 0.0)
            for trait in self._TRAIT_KEYWORDS
        }

        liked_ranked = sorted(net_scores.items(), key=lambda pair: pair[1], reverse=True)
        disliked_ranked = sorted(net_scores.items(), key=lambda pair: pair[1])

        liked_traits = [trait for trait, score in liked_ranked if score > 0][:limit]
        avoided_traits = [trait for trait, score in disliked_ranked if score < 0][:limit]

        # Fallback when only one side has positive signal in a round.
        if not liked_traits:
            liked_traits = self._top_traits(liked_scores, limit=limit)
        if not avoided_traits:
            avoided_traits = self._top_traits(disliked_scores, limit=limit)

        # Final guardrail: ensure no overlap in output lists.
        liked_set = set(liked_traits)
        avoided_traits = [trait for trait in avoided_traits if trait not in liked_set]
        return liked_traits, avoided_traits

    def _build_directive(self, traits: list[str], prefix: str) -> str:
        if not traits:
            return ""
        return f"{prefix} {'; '.join(traits)}."

    def _mean(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))
