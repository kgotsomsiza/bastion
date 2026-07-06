from __future__ import annotations

from frugalrouter.evaluation.verifier import LocalVerifier
from frugalrouter.model_policy import ModelPolicy
from frugalrouter.providers.fireworks import FireworksProvider
from frugalrouter.providers.local import LocalProvider
from frugalrouter.task_classifier import classify_prompt
from frugalrouter.types import RouteResult, Task, Verification


class FrugalRouter:
    def __init__(self, config: dict, allow_remote: bool = True) -> None:
        self.config = config
        self.allow_remote = allow_remote
        self.local_confidence_threshold = float(config.get("local_confidence_threshold", 0.92))
        self.local = LocalProvider()
        self.remote = FireworksProvider(config)
        self.model_policy = ModelPolicy(config)
        self.verifier = LocalVerifier()

    def run(self, task: Task) -> RouteResult:
        category = classify_prompt(task.input)
        local_candidate = self.local.answer(task)

        if local_candidate.confidence >= self.local_confidence_threshold:
            return RouteResult(
                task_id=task.id,
                answer=local_candidate.answer,
                verification=Verification(confidence=local_candidate.confidence, reasons=local_candidate.reasons),
                route="local",
                used_remote=False,
                category=category,
            )

        fallback_reason = (
            f"local_confidence_{local_candidate.confidence:.2f}_below_{self.local_confidence_threshold:.2f}"
        )
        if not self.allow_remote:
            return RouteResult(
                task_id=task.id,
                answer=local_candidate.answer,
                verification=Verification(confidence=local_candidate.confidence, reasons=local_candidate.reasons),
                route="local_remote_disabled",
                used_remote=False,
                fallback_reason=fallback_reason,
                category=category,
            )

        model = self.model_policy.choose(category)
        try:
            remote_answer = self.remote.answer(task, model=model, category=category)
        except Exception as error:
            return RouteResult(
                task_id=task.id,
                answer=local_candidate.answer,
                verification=Verification(confidence=local_candidate.confidence, reasons=local_candidate.reasons),
                route="remote_error",
                used_remote=False,
                fallback_reason=f"{fallback_reason};remote_error:{error}",
                category=category,
            )

        remote_verification = self.verifier.score(task, remote_answer)
        return RouteResult(
            task_id=task.id,
            answer=remote_answer,
            verification=remote_verification,
            route="remote",
            used_remote=True,
            fallback_reason=fallback_reason,
            category=category,
        )
