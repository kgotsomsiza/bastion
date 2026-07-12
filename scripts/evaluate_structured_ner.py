"""Audit deterministic structured NER against an exact-entity task set."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frugalrouter.providers.local import LocalProvider  # noqa: E402
from frugalrouter.types import Task  # noqa: E402


def _normalized(items: list[str]) -> Counter[str]:
    return Counter(" ".join(item.lower().split()) for item in items if item.strip())


def _gold(task: dict) -> list[str]:
    for key in ("expected_entity_set", "expected_contains_all"):
        values = task.get(key)
        if isinstance(values, list) and all(isinstance(value, str) for value in values):
            return values
    raise ValueError(f"{task.get('task_id', '<unknown>')} has no exact entity list")


def evaluate(tasks: list[dict]) -> tuple[dict, list[dict]]:
    provider = LocalProvider()
    rows: list[dict] = []
    for spec in tasks:
        prompt = str(spec["prompt"])
        candidate = provider.answer(Task(id=str(spec["task_id"]), input=prompt))
        structured = "computed_structured_ner" in candidate.reasons
        answer_items = [line.strip() for line in candidate.answer.text.splitlines() if line.strip()]
        expected = _gold(spec)
        correct = structured and _normalized(answer_items) == _normalized(expected)
        rows.append(
            {
                "task_id": spec["task_id"],
                "prompt": prompt,
                "expected": expected,
                "answer": candidate.answer.text,
                "accepted": structured,
                "correct": correct,
                "dangerous_accept": structured and not correct,
                "confidence": candidate.confidence,
                "reasons": candidate.reasons,
            }
        )

    accepted = sum(row["accepted"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    dangerous = sum(row["dangerous_accept"] for row in rows)
    summary = {
        "tasks": len(rows),
        "accepted": accepted,
        "accepted_correct": correct,
        "dangerous_accepts": dangerous,
        "coverage": accepted / len(rows) if rows else 0.0,
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    summary, rows = evaluate(tasks)
    report = {"summary": summary, "rows": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for row in rows:
        if row["dangerous_accept"]:
            print(json.dumps(row, ensure_ascii=False))
    return 1 if summary["dangerous_accepts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
