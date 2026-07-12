"""Measure local-model correctness against token-level confidence signals."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frugalrouter.eval_runner import grade_answer  # noqa: E402
from frugalrouter.local_verify import verify_local_answer  # noqa: E402
from frugalrouter.prompting import CATEGORY_INSTRUCTIONS, clean_answer  # noqa: E402
from frugalrouter.providers.local_model import (  # noqa: E402
    LOCAL_CATEGORY_INSTRUCTIONS,
    LOCAL_MODEL_SYSTEM,
)


def confidence_from_logits(scores: np.ndarray) -> tuple[float, float]:
    values = np.asarray(scores, dtype=np.float64)
    top_two = np.partition(values, -2)[-2:]
    maximum = float(top_two.max())
    second = float(top_two.min())
    log_probability = maximum - (maximum + math.log(float(np.exp(values - maximum).sum())))
    return math.exp(log_probability), maximum - second


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
    load_seconds = time.perf_counter() - started
    rows: list[dict[str, Any]] = []

    for index, spec in enumerate(tasks, 1):
        category = str(spec.get("category") or "general")
        instruction = LOCAL_CATEGORY_INSTRUCTIONS.get(
            category, CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS["general"])
        )
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
        grade = grade_answer(answer, spec)
        row = {
            "task_id": spec.get("task_id"),
            "category": category,
            "prompt": spec["prompt"],
            "answer": answer,
            "correct": bool(grade["passed"]),
            "grade_reason": grade["reason"],
            "verified": verify_local_answer(spec["prompt"], category, answer),
            "latency_seconds": round(latency, 3),
            "completion_tokens": completion_tokens,
            "finish_reason": choice.get("finish_reason"),
            **summarize_confidence(probabilities, margins),
        }
        rows.append(row)
        print(
            f"{index}/{len(tasks)} {row['task_id']} {category} correct={row['correct']} "
            f"pmean={row['mean_probability']:.3f} pmin={row['min_probability']:.3f} "
            f"{latency:.1f}s",
            flush=True,
        )

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
