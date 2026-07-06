from __future__ import annotations

import re
import time
from dataclasses import dataclass

from frugalrouter.types import Answer, Task


@dataclass(frozen=True)
class LocalCandidate:
    answer: Answer
    confidence: float
    reasons: list[str]


class LocalProvider:
    name = "local"
    model = "deterministic-shortcuts"

    def answer(self, task: Task) -> LocalCandidate:
        started = time.perf_counter()
        text, confidence, reasons = self._answer_text(task)
        latency_ms = int((time.perf_counter() - started) * 1000)
        answer = Answer(text=text, provider=self.name, model=self.model, latency_ms=latency_ms)
        return LocalCandidate(answer=answer, confidence=confidence, reasons=reasons)

    def _answer_text(self, task: Task) -> tuple[str, float, list[str]]:
        prompt = task.input.strip()
        lower = prompt.lower()

        if self._is_sentiment_prompt(lower):
            return self._sentiment(prompt)

        arithmetic = self._simple_arithmetic(prompt)
        if arithmetic is not None:
            return arithmetic

        return "", 0.0, ["no_local_shortcut"]

    def _sentiment(self, prompt: str) -> tuple[str, float, list[str]]:
        positive = {"good", "great", "fast", "clear", "useful", "love", "excellent", "happy", "impressed"}
        negative = {"bad", "slow", "confusing", "broken", "hate", "poor", "wrong", "sad", "terrible", "awful"}
        words = set(re.findall(r"[a-zA-Z']+", prompt.lower()))
        score = len(words & positive) - len(words & negative)
        if score > 0:
            return "positive", 0.94, ["clear_sentiment_keywords"]
        if score < 0:
            return "negative", 0.94, ["clear_sentiment_keywords"]
        return "neutral", 0.9, ["no_clear_sentiment_keywords"]

    def _simple_arithmetic(self, prompt: str) -> tuple[str, float, list[str]] | None:
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*([+*/-])\s*(-?\d+(?:\.\d+)?)", prompt)
        if not match:
            return None
        left = float(match.group(1))
        op = match.group(2)
        right = float(match.group(3))
        if op == "+":
            value = left + right
        elif op == "-":
            value = left - right
        elif op == "*":
            value = left * right
        elif op == "/" and right != 0:
            value = left / right
        else:
            return None
        if value.is_integer():
            return str(int(value)), 0.97, ["direct_arithmetic_expression"]
        return f"{value:.6g}", 0.96, ["direct_arithmetic_expression"]

    def _is_sentiment_prompt(self, prompt: str) -> bool:
        return "sentiment" in prompt and any(word in prompt for word in ["classify", "label", "positive", "negative"])
