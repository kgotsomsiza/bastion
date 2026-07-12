from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from frugalrouter.eval_runner import grade_answer  # noqa: E402
from frugalrouter.local_verify import verify_local_answer  # noqa: E402
from frugalrouter.prompting import CATEGORY_INSTRUCTIONS, clean_answer  # noqa: E402
from frugalrouter.providers.local_model import (  # noqa: E402
    LOCAL_CATEGORY_INSTRUCTIONS,
    LOCAL_MODEL_SYSTEM,
)


def call_server(prompt: str, category: str, port: int) -> tuple[str, int, float | None]:
    instruction = LOCAL_CATEGORY_INSTRUCTIONS.get(
        category,
        CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS["general"]),
    )
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": LOCAL_MODEL_SYSTEM},
                {"role": "user", "content": f"{instruction}\n{prompt}"},
            ],
            "temperature": 0.0,
            "max_tokens": 128,
            "logprobs": True,
            "top_logprobs": 5,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    latency_ms = round((time.perf_counter() - started) * 1000)
    choice = payload["choices"][0]
    raw = choice["message"]["content"] or ""
    token_rows = (choice.get("logprobs") or {}).get("content") or []
    first_logprob = float(token_rows[0]["logprob"]) if token_rows else None
    return clean_answer(raw, category), latency_ms, first_logprob


def _load_rows(path: str | Path) -> list[dict]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        payload = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"{source} must contain a JSON array or JSONL rows")

    specs = []
    for index, row in enumerate(payload, start=1):
        if "messages" not in row:
            specs.append(row)
            continue
        category = row["category"]
        user_message = next(message["content"] for message in row["messages"] if message["role"] == "user")
        prompt = user_message.split("\n", 1)[-1]
        gold = next(message["content"] for message in row["messages"] if message["role"] == "assistant").strip()
        spec = {
            "task_id": f"{row.get('family', source.stem)}-{index:04d}",
            "category": category,
            "prompt": prompt,
        }
        if category == "sentiment":
            spec["expected_label"] = gold
        elif category == "ner":
            spec["expected_entity_set"] = [item.strip() for item in gold.split(",") if item.strip()]
        else:
            spec["expected_exact"] = gold
        specs.append(spec)
    return specs


def _parse_ner_items(answer: str) -> list[str]:
    items: list[str] = []
    for line in answer.splitlines():
        line = re.sub(r"^[\s*\u2022-]*(?:[A-Za-z ]{2,20}:)?", "", line).strip()
        if line:
            items.extend(part.strip(" .;") for part in re.split(r"[,;]| and ", line) if part.strip(" .;"))
    return items


def _grade(answer: str, spec: dict) -> dict:
    if "expected_entity_set" not in spec:
        return grade_answer(answer, spec)
    actual = Counter(" ".join(item.lower().split()) for item in _parse_ner_items(answer))
    expected = Counter(" ".join(item.lower().split()) for item in spec["expected_entity_set"])
    return {
        "passed": actual == expected,
        "reason": f"entity_set:expected={dict(expected)};actual={dict(actual)}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--categories", nargs="*", default=[])
    parser.add_argument("--model-label", default="local-model")
    args = parser.parse_args()

    specs: list[dict] = []
    for task_file in args.tasks:
        specs.extend(_load_rows(task_file))
    if args.categories:
        allowed = set(args.categories)
        specs = [spec for spec in specs if spec["category"] in allowed]

    rows = []
    for index, spec in enumerate(specs, start=1):
        prompt = spec.get("prompt") or spec.get("input")
        category = spec["category"]
        answer, latency_ms, first_token_logprob = call_server(prompt, category, args.port)
        accepted = verify_local_answer(prompt, category, answer)
        grade = _grade(answer, spec)
        passed = bool(grade["passed"])
        row = {
            "task_id": spec.get("task_id") or spec.get("id"),
            "category": category,
            "answer": answer,
            "accepted": accepted,
            "passed": passed,
            "dangerous_wrong_accept": accepted and not passed,
            "grade_reason": grade["reason"],
            "latency_ms": latency_ms,
            "first_token_logprob": first_token_logprob,
            "first_token_probability": math.exp(first_token_logprob) if first_token_logprob is not None else None,
            "model_label": args.model_label,
        }
        rows.append(row)
        print(
            f"[{index:03d}/{len(specs)}] {row['task_id']}: "
            f"accepted={accepted} passed={passed} latency_ms={latency_ms} answer={answer!r}",
            flush=True,
        )

    accepted = [row for row in rows if row["accepted"]]
    dangerous = [row for row in rows if row["dangerous_wrong_accept"]]
    summary = {
        "model_label": args.model_label,
        "tasks": len(rows),
        "raw_correct": sum(row["passed"] for row in rows),
        "accepted": len(accepted),
        "accepted_correct": sum(row["passed"] for row in accepted),
        "dangerous_wrong_accepts": len(dangerous),
        "dangerous_task_ids": [row["task_id"] for row in dangerous],
        "median_latency_ms": sorted(row["latency_ms"] for row in rows)[len(rows) // 2] if rows else None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    raise SystemExit(2 if dangerous else 0)


if __name__ == "__main__":
    main()
