from __future__ import annotations

import re

from frugalrouter.types import Answer, Task, Verification


class LocalVerifier:
    def score(self, task: Task, answer: Answer) -> Verification:
        confidence = 0.5
        reasons: list[str] = []
        text = answer.text.strip()

        if not text:
            return Verification(confidence=0.0, reasons=["empty_answer"])

        if task.expected_format == "label":
            allowed = {"positive", "neutral", "negative"}
            if text.lower() in allowed:
                confidence += 0.3
                reasons.append("valid_label")
            else:
                confidence -= 0.35
                reasons.append("invalid_label")

        if task.expected_format == "short_text":
            word_count = len(re.findall(r"\S+", text))
            if word_count <= 12:
                confidence += 0.2
                reasons.append("short_answer")
            else:
                confidence -= 0.2
                reasons.append("too_long")

        if self._looks_uncertain(text):
            confidence -= 0.25
            reasons.append("uncertain_language")

        if answer.provider == "local":
            confidence -= self._difficulty_penalty(task)

        return Verification(confidence=max(0.0, min(1.0, confidence)), reasons=reasons)

    def _difficulty_penalty(self, task: Task) -> float:
        prompt = task.input.lower()
        hard_markers = [
            "calculate",
            "prove",
            "reason step by step",
            "code",
            "write a function",
            "extract all",
            "compare",
            "multimodal",
        ]
        return 0.25 if any(marker in prompt for marker in hard_markers) else 0.0

    def _looks_uncertain(self, text: str) -> bool:
        lower = text.lower()
        return any(marker in lower for marker in ["i don't know", "maybe", "not sure", "cannot determine"])

