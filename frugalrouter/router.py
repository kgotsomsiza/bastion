from __future__ import annotations

from frugalrouter.evaluation.verifier import LocalVerifier
from frugalrouter.providers.fireworks import FireworksProvider
from frugalrouter.providers.local import LocalProvider
from frugalrouter.types import RouteResult, Task


class FrugalRouter:
    def __init__(self, config: dict, allow_remote: bool = False) -> None:
        self.config = config
        self.allow_remote = allow_remote
        self.remote_threshold = float(config.get("remote_threshold", 0.72))
        self.local = LocalProvider()
        self.remote = FireworksProvider(config)
        self.verifier = LocalVerifier()

    def run(self, task: Task) -> RouteResult:
        local_answer = self.local.answer(task)
        local_verification = self.verifier.score(task, local_answer)

        if local_verification.confidence >= self.remote_threshold:
            return RouteResult(
                task_id=task.id,
                answer=local_answer,
                verification=local_verification,
                route="local",
                used_remote=False,
            )

        fallback_reason = f"local_confidence_{local_verification.confidence:.2f}_below_{self.remote_threshold:.2f}"
        if not self.allow_remote:
            return RouteResult(
                task_id=task.id,
                answer=local_answer,
                verification=local_verification,
                route="local_remote_disabled",
                used_remote=False,
                fallback_reason=fallback_reason,
            )

        remote_answer = self.remote.answer(task)
        remote_verification = self.verifier.score(task, remote_answer)
        return RouteResult(
            task_id=task.id,
            answer=remote_answer,
            verification=remote_verification,
            route="remote",
            used_remote=True,
            fallback_reason=fallback_reason,
        )

