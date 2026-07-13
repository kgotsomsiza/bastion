"""Measure local-model correctness against token-level confidence signals."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frugalrouter.eval_runner import grade_answer  # noqa: E402
from frugalrouter.local_verify import verify_local_answer  # noqa: E402
from frugalrouter.prompting import clean_answer, prompt_wants_explanation  # noqa: E402
from frugalrouter.providers.local_model import (  # noqa: E402
    LOCAL_MODEL_SYSTEM,
    confidence_from_logits,
    configure_non_thinking_chat,
    instruction_for_local_task,
)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def summarize_confidence(probabilities: list[float], margins: list[float]) -> dict[str, float]:
    if not probabilities:
        return {"first_probability": 0.0, "min_probability": 0.0, "mean_probability": 0.0,
                "geometric_mean_probability": 0.0, "mean_margin": 0.0}
    return {
        "first_probability": probabilities[0],
        "min_probability": min(probabilities),
        "mean_probability": sum(probabilities) / len(probabilities),
        "geometric_mean_probability": math.exp(
            sum(math.log(max(value, 1e-300)) for value in probabilities) / len(probabilities)
        ),
        "mean_margin": sum(margins) / len(margins),
    }


def threshold_sweep(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    results = []
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99):
        accepted = [row for row in rows if row[metric] >= threshold]
        correct = [row for row in accepted if row["correct"]]
        results.append({
            "metric": metric,
            "threshold": threshold,
            "accepted": len(accepted),
            "correct": len(correct),
            "wrong": len(accepted) - len(correct),
            "accuracy": len(correct) / len(accepted) if accepted else None,
        })
    return results


def checkpoint_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".rows.jsonl")


def load_checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_ner_items(answer: str) -> list[str]:
    text = (answer or "").strip()
    if text.startswith(("[", "{")):
        try:
            structured = json.loads(text)
        except json.JSONDecodeError:
            structured = None
        if isinstance(structured, list) and all(isinstance(item, str) for item in structured):
            return [item.strip() for item in structured if item.strip()]
        if isinstance(structured, dict):
            values: list[str] = []
            for value in structured.values():
                if isinstance(value, str) and value.strip():
                    values.append(value.strip())
                elif isinstance(value, list):
                    values.extend(str(item).strip() for item in value if str(item).strip())
            if values:
                return values

    items: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^[\s*\u2022-]*(?:[A-Za-z ]{2,20}:)?", "", line).strip()
        if line:
            items.extend(
                part.strip(" .;\"'")
                for part in re.split(r"[,;|]|\s+and\s+", line)
                if part.strip(" .;\"'")
            )
    return items


def _normalized_counter(items: list[Any]) -> Counter[str]:
    return Counter(" ".join(str(item).lower().split()) for item in items)


def grade_task(answer: str, spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("category") == "ner":
        expected = spec.get("expected_entity_set")
        if expected is None:
            expected = spec.get("expected_contains_all")
        if expected is not None:
            actual_counter = _normalized_counter(parse_ner_items(answer))
            expected_counter = _normalized_counter(list(expected))
            return {
                "passed": actual_counter == expected_counter,
                "reason": f"exact_entity_multiset:{dict(expected_counter)}",
            }
    return grade_answer(answer, spec)


def run(model_path: Path, tasks_path: Path, output_path: Path) -> int:
    from llama_cpp import Llama, LogitsProcessorList

    tasks = json.loads(tasks_path.read_text(encoding="utf-8-sig"))
    started = time.perf_counter()
    model = Llama(
        model_path=str(model_path),
        n_ctx=int(os.getenv("BAKEOFF_N_CTX", "2048")),
        n_threads=int(os.getenv("BAKEOFF_N_THREADS", "2")),
        verbose=False,
    )
    disable_thinking_requested = env_flag("BAKEOFF_DISABLE_THINKING")
    non_thinking_configured = (
        configure_non_thinking_chat(model) if disable_thinking_requested else False
    )
    if disable_thinking_requested and not non_thinking_configured:
        raise RuntimeError("BAKEOFF_DISABLE_THINKING was requested but no GGUF chat template exists")
    load_seconds = time.perf_counter() - started
    checkpoint_path = checkpoint_path_for(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if env_flag("BAKEOFF_RESUME"):
        rows = load_checkpoint_rows(checkpoint_path)
    else:
        checkpoint_path.unlink(missing_ok=True)
        rows = []
    completed_task_ids = {str(row["task_id"]) for row in rows}
    if rows:
        print(f"resuming with {len(rows)} checkpointed tasks", flush=True)

    for index, spec in enumerate(tasks, 1):
        task_id = str(spec.get("task_id") or f"task-{index}")
        if task_id in completed_task_ids:
            continue
        category = str(spec.get("category") or "general")
        explanation_requested = prompt_wants_explanation(spec["prompt"])
        instruction = instruction_for_local_task(category, spec["prompt"])
        probabilities: list[float] = []
        margins: list[float] = []

        def capture(_input_ids, scores):
            probability, margin = confidence_from_logits(scores)
            probabilities.append(probability)
            margins.append(margin)
            return scores

        task_started = time.perf_counter()
        response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": LOCAL_MODEL_SYSTEM},
                {"role": "user", "content": f"{instruction}\n{spec['prompt']}"},
            ],
            temperature=0.0,
            max_tokens=int(os.getenv("BAKEOFF_MAX_TOKENS", "128")),
            logits_processor=LogitsProcessorList([capture]),
        )
        latency = time.perf_counter() - task_started
        choice = response["choices"][0]
        answer = clean_answer((choice.get("message") or {}).get("content") or "", category, prompt=spec["prompt"])
        completion_tokens = int((response.get("usage") or {}).get("completion_tokens", len(probabilities)))
        probabilities = probabilities[:completion_tokens]
        margins = margins[:completion_tokens]
        grade = grade_task(answer, spec)
        row = {
            "task_id": task_id,
            "category": category,
            "prompt": spec["prompt"],
            "answer": answer,
            "correct": bool(grade["passed"]),
            "grade_reason": grade["reason"],
            "explanation_requested": explanation_requested,
            "verified": verify_local_answer(spec["prompt"], category, answer),
            "latency_seconds": round(latency, 3),
            "completion_tokens": completion_tokens,
            "finish_reason": choice.get("finish_reason"),
            **summarize_confidence(probabilities, margins),
        }
        rows.append(row)
        with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
            checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
            checkpoint.flush()
            os.fsync(checkpoint.fileno())
        print(
            f"{index}/{len(tasks)} {row['task_id']} {category} correct={row['correct']} "
            f"pmean={row['mean_probability']:.3f} pmin={row['min_probability']:.3f} "
            f"{latency:.1f}s",
            flush=True,
        )

    rows_by_id = {str(row["task_id"]): row for row in rows}
    rows = [
        rows_by_id[task_id]
        for index, spec in enumerate(tasks, 1)
        if (task_id := str(spec.get("task_id") or f"task-{index}")) in rows_by_id
    ]

    categories: dict[str, dict[str, Any]] = {}
    for category in sorted({row["category"] for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        categories[category] = {
            "tasks": len(subset),
            "correct": sum(row["correct"] for row in subset),
            "verified": sum(row["verified"] for row in subset),
            "verified_wrong": sum(row["verified"] and not row["correct"] for row in subset),
            "sweeps": threshold_sweep(subset, "geometric_mean_probability"),
        }
    report = {
        "model": str(model_path),
        "tasks_file": str(tasks_path),
        "disable_thinking_requested": disable_thinking_requested,
        "non_thinking_configured": non_thinking_configured,
        "load_seconds": round(load_seconds, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "summary": {
            "tasks": len(rows),
            "correct": sum(row["correct"] for row in rows),
            "verified": sum(row["verified"] for row in rows),
            "verified_wrong": sum(row["verified"] and not row["correct"] for row in rows),
        },
        "categories": categories,
        "rows": rows,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    return run(args.model, args.tasks, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
