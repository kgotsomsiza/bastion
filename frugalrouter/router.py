from __future__ import annotations

import time

from frugalrouter.evaluation.verifier import LocalVerifier
from frugalrouter.model_policy import ModelPolicy
from frugalrouter.providers.fireworks import FireworksError, FireworksProvider
from frugalrouter.providers.local import LocalProvider
from frugalrouter.task_classifier import classify_prompt
from frugalrouter.types import RouteResult, Task, Verification


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class FrugalRouter:
    def __init__(self, config: dict, allow_remote: bool = True) -> None:
        self.config = config
        self.allow_remote = allow_remote
        self.local_confidence_threshold = float(config.get("local_confidence_threshold", 0.92))
        self.remote_max_attempts = int(config.get("remote_max_attempts", 3))
        self.remote_backoff_seconds = float(config.get("remote_backoff_seconds", 2.0))
        self.rescue_max_tokens = int(config.get("rescue_max_tokens", 900))
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

        errors: list[str] = []
        remote_answer = None
        for model in self.model_policy.candidates(category):
            remote_answer, abort = self._call_with_retry(task, model, category, errors)
            if remote_answer is not None or abort:
                break

        if remote_answer is None:
            return RouteResult(
                task_id=task.id,
                answer=local_candidate.answer,
                verification=Verification(confidence=local_candidate.confidence, reasons=local_candidate.reasons),
                route="remote_error",
                used_remote=False,
                fallback_reason=f"{fallback_reason};remote_error:{' | '.join(errors)}",
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

    def _call_with_retry(self, task: Task, model: str, category: str, errors: list[str]):
        """Call one model with backoff on transient errors.

        Returns (answer, abort). answer is None when this model failed;
        abort=True means the error is not worth trying other models for.
        """
        max_tokens_override: int | None = None
        for attempt in range(self.remote_max_attempts):
            kwargs = {"max_tokens_override": max_tokens_override} if max_tokens_override else {}
            try:
                answer = self.remote.answer(task, model=model, category=category, **kwargs)
            except FireworksError as error:
                errors.append(f"{model}:{error}")
                if error.status_code == 404:
                    return None, False
                if error.status_code in RETRYABLE_STATUS_CODES:
                    if attempt < self.remote_max_attempts - 1:
                        time.sleep(self.remote_backoff_seconds * (2**attempt))
                    continue
                return None, True
            except Exception as error:  # noqa: BLE001 - any provider bug must not kill the batch
                errors.append(f"{model}:{error}")
                return None, True

            # Reasoning-heavy models can spend the whole cap before emitting
            # an answer; an empty submission is a guaranteed zero, so retry
            # once with a generous cap.
            if not answer.text.strip() and answer.finish_reason == "length" and max_tokens_override is None:
                max_tokens_override = self.rescue_max_tokens
                errors.append(f"{model}:empty_at_cap_retrying_with_{max_tokens_override}")
                continue
            return answer, False
        return None, False
