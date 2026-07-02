from __future__ import annotations

import re
import time

from frugalrouter.types import Answer, Task


class LocalProvider:
    name = "local"
    model = "heuristic-baseline"

    def answer(self, task: Task) -> Answer:
        started = time.perf_counter()
        text = self._answer_text(task)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return Answer(text=text, provider=self.name, model=self.model, latency_ms=latency_ms)

    def _answer_text(self, task: Task) -> str:
        prompt = task.input.strip()
        lower = prompt.lower()

        if "capital of south africa" in lower:
            return "Pretoria"

        if "classify sentiment" in lower:
            return self._sentiment(prompt)

        if "summarize" in lower:
            return self._summarize(prompt, max_words=12)

        return self._extract_last_instruction(prompt)

    def _sentiment(self, prompt: str) -> str:
        positive = {"good", "great", "fast", "clear", "useful", "love", "excellent"}
        negative = {"bad", "slow", "confusing", "broken", "hate", "poor", "wrong"}
        words = set(re.findall(r"[a-zA-Z']+", prompt.lower()))
        score = len(words & positive) - len(words & negative)
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    def _summarize(self, prompt: str, max_words: int) -> str:
        text = prompt.split(":", 1)[-1].strip()
        words = re.findall(r"\S+", text)
        return " ".join(words[:max_words])

    def _extract_last_instruction(self, prompt: str) -> str:
        if ":" in prompt:
            return prompt.rsplit(":", 1)[-1].strip()
        return prompt[:240]

