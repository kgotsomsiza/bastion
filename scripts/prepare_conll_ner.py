"""Prepare disjoint, human-labeled CoNLL-2003 NER train/test artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from frugalrouter.local_verify import verify_local_answer
from frugalrouter.prompting import CATEGORY_INSTRUCTIONS
from frugalrouter.providers.local_model import LOCAL_MODEL_SYSTEM


KINDS = {
    "PER": "person names",
    "ORG": "organization names",
    "LOC": "location names",
}


def _detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([($])\s+", r"\1", text)
    text = re.sub(r"\s+(['’](?:s|re|ve|ll|d|m|t))\b", r"\1", text, flags=re.IGNORECASE)
    return text.strip()


def _extract_spans(tokens: list[str], tag_ids: list[int], tag_names: list[str], target: str) -> list[str]:
    spans: list[list[str]] = []
    current: list[str] = []
    for token, tag_id in zip(tokens, tag_ids, strict=True):
        tag = tag_names[int(tag_id)]
        if tag == f"B-{target}":
            if current:
                spans.append(current)
            current = [token]
        elif tag == f"I-{target}" and current:
            current.append(token)
        else:
            if current:
                spans.append(current)
                current = []
    if current:
        spans.append(current)
    return [_detokenize(span) for span in spans]


def _candidates(dataset, tag_names: list[str], excluded_sources: set[str] | None = None) -> list[dict]:
    excluded_sources = excluded_sources or set()
    rows = []
    seen = set()
    for row in dataset:
        tokens = list(row["tokens"])
        if not 5 <= len(tokens) <= 60:
            continue
        source = _detokenize(tokens)
        source_key = " ".join(source.lower().split())
        if source_key in excluded_sources:
            continue
        for tag, kind in KINDS.items():
            spans = _extract_spans(tokens, list(row["ner_tags"]), tag_names, tag)
            if not spans or any("," in span or ";" in span or " and " in span.lower() for span in spans):
                continue
            prompt = f"Extract all {kind} from this text: {source}"
            key = (prompt.lower(), tuple(span.lower() for span in spans))
            if key in seen or not verify_local_answer(prompt, "ner", ", ".join(spans)):
                continue
            seen.add(key)
            rows.append({"source_key": source_key, "kind": tag, "prompt": prompt, "spans": spans})
    return rows


def _select_balanced(rows: list[dict], per_kind: int, seed: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["kind"]].append(row)
    selected = []
    used_sources = set()
    for offset, kind in enumerate(KINDS):
        candidates = list(buckets[kind])
        random.Random(seed + offset).shuffle(candidates)
        for row in candidates:
            if row["source_key"] in used_sources:
                continue
            selected.append(row)
            used_sources.add(row["source_key"])
            if sum(item["kind"] == kind for item in selected) == per_kind:
                break
        if sum(item["kind"] == kind for item in selected) != per_kind:
            raise RuntimeError(f"Could not select {per_kind} unique-source {kind} tasks")
    random.Random(seed).shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-kind", type=int, default=150)
    parser.add_argument("--test-per-kind", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset("lhoestq/conll2003")
    tag_names = list(dataset["test"].features["ner_tags"].feature.names)
    test_rows = _select_balanced(_candidates(dataset["test"], tag_names), args.test_per_kind, args.seed)
    test_sources = {row["source_key"] for row in test_rows}
    train_rows = _select_balanced(
        _candidates(dataset["train"], tag_names, excluded_sources=test_sources),
        args.train_per_kind,
        args.seed + 100,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_path = args.output_dir / "conll_ner_test.json"
    train_path = args.output_dir / "conll_ner_train.jsonl"
    test_specs = [
        {
            "task_id": f"conll-test-{index:04d}",
            "category": "ner",
            "prompt": row["prompt"],
            "expected_entity_set": row["spans"],
        }
        for index, row in enumerate(test_rows, start=1)
    ]
    test_path.write_text(json.dumps(test_specs, ensure_ascii=False, indent=2), encoding="utf-8")

    instruction = CATEGORY_INSTRUCTIONS["ner"]
    with train_path.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            payload = {
                "messages": [
                    {"role": "system", "content": LOCAL_MODEL_SYSTEM},
                    {"role": "user", "content": f"{instruction}\n{row['prompt']}"},
                    {"role": "assistant", "content": ", ".join(row["spans"])},
                ]
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    for path in (train_path, test_path):
        print(f"{path}: sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
    train_sources = {row["source_key"] for row in train_rows}
    print(f"train={len(train_rows)} test={len(test_rows)} source_overlap={len(test_sources & train_sources)}")


if __name__ == "__main__":
    main()
