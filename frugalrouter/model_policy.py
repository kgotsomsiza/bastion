from __future__ import annotations

from typing import Any


DEFAULT_ALLOWED_MODELS = [
    "minimax-m3",
    "kimi-k2p7-code",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it-nvfp4",
]


class ModelPolicy:
    def __init__(self, config: dict[str, Any]) -> None:
        self.allowed_models = config.get("allowed_models") or DEFAULT_ALLOWED_MODELS
        self.policy = config.get("model_policy", {})
        self.default_model = config.get("fireworks", {}).get("default_model")

    def choose(self, category: str) -> str:
        return self.candidates(category)[0]

    def candidates(self, category: str) -> list[str]:
        preferences = self.policy.get(category) or self.policy.get("general") or []
        if self.default_model:
            preferences = [*preferences, self.default_model]
        preferences = [*preferences, *DEFAULT_ALLOWED_MODELS]

        candidates: list[str] = []
        for preference in preferences:
            match = self._find_allowed(preference)
            if match and match not in candidates:
                candidates.append(match)

        if not self.allowed_models:
            raise RuntimeError("No allowed Fireworks models are configured.")
        for model in self.allowed_models:
            if model not in candidates:
                candidates.append(model)
        return candidates

    def _find_allowed(self, preference: str) -> str | None:
        preferred = preference.lower()
        for model in self.allowed_models:
            candidate = model.lower()
            if candidate == preferred or preferred in candidate:
                return model
        return None
