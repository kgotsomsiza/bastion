"""Prepare disjoint, human-labeled TweetEval sentiment train/test artifacts.

The official test split is fixed before any subsequent fine-tuning and is
never included in training. Examples that Bastion's runtime nuance gate would
reject are excluded because the local model would never be trusted on them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from frugalrouter.local_verify import verify_local_answer
from frugalrouter.providers.local_model import LOCAL_CATEGORY_INSTRUCTIONS, LOCAL_MODEL_SYSTEM


ASK = "Classify the sentiment as positive, negative, or neutral"
LABELS = ("negative", "neutral", "positive")


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def _eligible_rows(dataset, label_names: list[str], excluded: set[str] | None = None) -> list[dict]:
    excluded = excluded or set()
    rows = []
    seen = set(excluded)
    for row in dataset:
        text = " ".join(str(row["text"]).split())
        normalized = _normalized(text)
        label = label_names[int(row["label"])]
        prompt = f"{ASK}: {text}"
        if normalized in seen or label not in LABELS:
            continue
        if not verify_local_answer(prompt, "sentiment", label):
            continue
        seen.add(normalized)
        rows.append({"text": text, "prompt": prompt, "label": label, "normalized": normalized})
    return rows


def _select_balanced(rows: list[dict], per_label: int, seed: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["label"]].append(row)
    selected = []
    for offset, label in enumerate(LABELS):
        rng = random.Random(seed + offset)
        rng.shuffle(buckets[label])
        if len(buckets[label]) < per_label:
            raise RuntimeError(f"Only {len(buckets[label])}/{per_label} eligible {label} rows")
        selected.extend(buckets[label][:per_label])
    random.Random(seed).shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-label", type=int, default=300)
    parser.add_argument("--test-per-label", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    label_names = list(dataset["test"].features["label"].names)

    test_rows = _select_balanced(
        _eligible_rows(dataset["test"], label_names),
        args.test_per_label,
        args.seed,
    )
    test_texts = {row["normalized"] for row in test_rows}
    train_rows = _select_balanced(
        _eligible_rows(dataset["train"], label_names, excluded=test_texts),
        args.train_per_label,
        args.seed + 100,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_path = args.output_dir / "tweet_eval_test.json"
    train_path = args.output_dir / "tweet_eval_train.jsonl"

    test_specs = [
        {
            "task_id": f"tweet-test-{index:04d}",
            "category": "sentiment",
            "prompt": row["prompt"],
            "expected_label": row["label"],
        }
        for index, row in enumerate(test_rows, start=1)
    ]
    test_path.write_text(json.dumps(test_specs, ensure_ascii=False, indent=2), encoding="utf-8")

    instruction = LOCAL_CATEGORY_INSTRUCTIONS["sentiment"]
    with train_path.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            payload = {
                "messages": [
                    {"role": "system", "content": LOCAL_MODEL_SYSTEM},
                    {"role": "user", "content": f"{instruction}\n{row['prompt']}"},
                    {"role": "assistant", "content": row["label"]},
                ]
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    for path in (train_path, test_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path}: sha256={digest}")
    print(f"train={len(train_rows)} test={len(test_rows)} overlap={len(test_texts & {row['normalized'] for row in train_rows})}")


if __name__ == "__main__":
    main()
