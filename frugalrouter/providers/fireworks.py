from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from frugalrouter.types import Answer, Task


class FireworksProvider:
    name = "fireworks"

    def __init__(self, config: dict[str, Any]) -> None:
        fireworks_config = config.get("fireworks", {})
        self.base_url = fireworks_config.get("base_url", "https://api.fireworks.ai/inference/v1/chat/completions")
        self.model = fireworks_config.get("model_id", "accounts/fireworks/models/gemma-7b-it")
        self.temperature = float(fireworks_config.get("temperature", 0.0))
        self.max_tokens = int(fireworks_config.get("max_tokens", 256))
        self.api_key = os.getenv("FIREWORKS_API_KEY")

    def answer(self, task: Task) -> Answer:
        if not self.api_key:
            raise RuntimeError("FIREWORKS_API_KEY is not set.")

        prompt = self._build_prompt(task)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer the task exactly. Keep the response minimal and do not add explanations.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Fireworks HTTP {error.code}: {detail}") from error

        latency_ms = int((time.perf_counter() - started) * 1000)
        choice = body["choices"][0]["message"]["content"].strip()
        usage = body.get("usage", {})
        return Answer(
            text=choice,
            provider=self.name,
            model=self.model,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
        )

    def _build_prompt(self, task: Task) -> str:
        format_hint = f"\nExpected format: {task.expected_format}" if task.expected_format else ""
        return f"{task.input}{format_hint}\nAnswer:"

