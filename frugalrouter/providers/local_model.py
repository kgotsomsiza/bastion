from __future__ import annotations

from dataclasses import dataclass
import math
import os
import threading
import time
from typing import Any

from frugalrouter.prompting import CATEGORY_INSTRUCTIONS, clean_answer, prompt_wants_explanation
from frugalrouter.types import Answer, Task


LOCAL_MODEL_SYSTEM = (
    "You are a precise assistant. Answer directly and minimally, with no preamble, "
    "no explanation unless asked, and no restating of the question."
)

# Constraint-explicit instructions raise the verification acceptance rate:
# every rejected local answer still costs a remote call, so compliant output
# is what makes this tier pay for itself.
LOCAL_CATEGORY_INSTRUCTIONS = {
    "sentiment": (
        "Respond with exactly one word - positive, negative, or neutral - and nothing else."
    ),
    "summarization": (
        "Write the summary in your own compressed words (do not copy a sentence verbatim). "
        "Obey any stated word or sentence limit exactly. Output only the summary."
    ),
}


@dataclass(frozen=True)
class LocalModelResult:
    answer: Answer
    first_probability: float
    min_probability: float
    geometric_mean_probability: float
    mean_margin: float


def instruction_for_local_task(category: str, prompt: str) -> str:
    if category in {"math", "logic"} and not prompt_wants_explanation(prompt):
        return (
            "Solve internally. Output only the requested final answer, with no reasoning, "
            "work, preamble, or FINAL ANSWER label."
        )
    return LOCAL_CATEGORY_INSTRUCTIONS.get(
        category, CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS["general"])
    )


def confidence_from_logits(scores: Any) -> tuple[float, float]:
    # NumPy is installed with llama-cpp-python in the local-model image, but
    # imported lazily so the lightweight V20 image can import this module.
    import numpy as np

    values = np.asarray(scores, dtype=np.float64)
    top_two = np.partition(values, -2)[-2:]
    maximum = float(top_two.max())
    second = float(top_two.min())
    probability = 1.0 / float(np.exp(values - maximum).sum())
    return probability, maximum - second


def _non_thinking_template(template: str) -> str:
    return "{%- set enable_thinking = false -%}\n" + template


def configure_non_thinking_chat(model: Any) -> bool:
    template = model.metadata.get("tokenizer.chat_template")
    if not template:
        return False

    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    eos_token_id = model.token_eos()
    bos_token_id = model.token_bos()
    eos_token = model._model.token_get_text(eos_token_id) if eos_token_id != -1 else ""
    bos_token = model._model.token_get_text(bos_token_id) if bos_token_id != -1 else ""
    model.chat_handler = Jinja2ChatFormatter(
        template=_non_thinking_template(template),
        eos_token=eos_token,
        bos_token=bos_token,
        stop_token_ids=[eos_token_id] if eos_token_id != -1 else None,
    ).to_chat_handler()
    model.chat_format = None
    return True


class LocalModelProvider:
    """A small GGUF model run on-CPU via llama-cpp-python.

    Answers count as ZERO Fireworks tokens (guide: local inference is free),
    so routing easy tasks here is pure token-ranking win. The provider is
    designed to degrade gracefully: if the weights are missing or llama_cpp
    is not installed, ``available_for`` returns False and the router falls
    through to Fireworks exactly as the baseline does. This is why bundling
    the model is optional and the baseline image is unaffected.
    """

    name = "local_model"

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("local_model", {})
        # Env overrides config so the same image behaves as baseline when
        # LOCAL_MODEL_PATH is unset.
        self.model_path = os.getenv("LOCAL_MODEL_PATH", cfg.get("model_path", "")).strip()
        self.model_label = cfg.get("model_label", "local-gguf")
        self.categories = set(cfg.get("categories", []))
        self.max_tokens = int(cfg.get("max_tokens", 256))
        self.n_ctx = int(cfg.get("n_ctx", 2048))
        self.n_threads = int(cfg.get("n_threads", os.cpu_count() or 2))
        self.n_batch = int(cfg.get("n_batch", 512))
        self.disable_thinking = bool(cfg.get("disable_thinking", False))
        self.confidence_thresholds = {
            str(category): float(threshold)
            for category, threshold in cfg.get("confidence_thresholds", {}).items()
        }
        self._llm: Any = None
        self._load_failed = False
        # llama.cpp is not thread-safe; the CLI runs tasks in parallel workers,
        # so all load + inference on the shared model must be serialized or it
        # segfaults. Fireworks calls don't take this lock and stay parallel.
        self._lock = threading.Lock()

    def available_for(self, category: str) -> bool:
        return category in self.categories and self._ensure_loaded()

    def _ensure_loaded(self) -> bool:
        if self._llm is not None:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._llm is not None:
                return True
            if self._load_failed or not self.model_path or not os.path.exists(self.model_path):
                self._load_failed = True
                return False
            try:
                from llama_cpp import Llama

                self._llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_batch=self.n_batch,
                    verbose=False,
                )
                if self.disable_thinking and not configure_non_thinking_chat(self._llm):
                    raise RuntimeError("local model has no GGUF chat template for non-thinking mode")
                return True
            except Exception:  # noqa: BLE001 - any load failure => fall back to remote
                self._load_failed = True
                return False

    def confidence_threshold_for(self, category: str) -> float:
        return self.confidence_thresholds.get(category, 1.01)

    def answer(
        self,
        task: Task,
        category: str = "general",
        corrective_hint: str | None = None,
    ) -> LocalModelResult:
        from llama_cpp import LogitsProcessorList

        started = time.perf_counter()
        instruction = instruction_for_local_task(category, task.input)
        if corrective_hint:
            instruction = f"{instruction}\n{corrective_hint}"
        probabilities: list[float] = []
        margins: list[float] = []

        def capture(_input_ids, scores):
            probability, margin = confidence_from_logits(scores)
            probabilities.append(probability)
            margins.append(margin)
            return scores

        with self._lock:
            completion = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": LOCAL_MODEL_SYSTEM},
                    {"role": "user", "content": f"{instruction}\n{task.input}"},
                ],
                temperature=0.0,
                max_tokens=self.max_tokens,
                logits_processor=LogitsProcessorList([capture]),
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        choice = completion["choices"][0]
        text = clean_answer(
            (choice.get("message") or {}).get("content") or "",
            category,
            prompt=task.input,
        )
        completion_tokens = int((completion.get("usage") or {}).get("completion_tokens", len(probabilities)))
        probabilities = probabilities[:completion_tokens]
        margins = margins[:completion_tokens]
        if probabilities:
            first_probability = probabilities[0]
            min_probability = min(probabilities)
            geometric_mean_probability = math.exp(
                sum(math.log(max(value, 1e-300)) for value in probabilities) / len(probabilities)
            )
            mean_margin = sum(margins) / len(margins)
        else:
            first_probability = min_probability = geometric_mean_probability = mean_margin = 0.0
        # prompt/completion tokens stay 0: local inference is free for scoring.
        return LocalModelResult(
            answer=Answer(
                text=text,
                provider=self.name,
                model=self.model_label,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency_ms,
                finish_reason=choice.get("finish_reason"),
            ),
            first_probability=first_probability,
            min_probability=min_probability,
            geometric_mean_probability=geometric_mean_probability,
            mean_margin=mean_margin,
        )
