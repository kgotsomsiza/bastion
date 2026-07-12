"""V21 model bake-off: measure a GGUF as the local tier under judge-box limits.

For each category subset: raw accuracy, verifier-accepted count, accepted-correct
(the ONLY number that gates shipping), and per-task latency. Uses the exact
runtime prompt (LocalModelProvider) and the STRICT V17 verifier.

Usage: model_bakeoff.py <model.gguf> <label> <cat1.json> [cat2.json ...]
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/root/bastion")
from frugalrouter.providers.local_model import LOCAL_MODEL_SYSTEM, LOCAL_CATEGORY_INSTRUCTIONS  # noqa: E402
from frugalrouter.prompting import CATEGORY_INSTRUCTIONS, clean_answer  # noqa: E402
from frugalrouter.local_verify import verify_local_answer  # noqa: E402


def parse_ner_items(answer: str) -> list[str]:
    items: list[str] = []
    for line in (answer or "").splitlines():
        line = re.sub(r"^[\s*\u2022-]*(?:[A-Za-z ]{2,20}:)?", "", line).strip()
        if line:
            items.extend(part.strip(" .;") for part in re.split(r"[,;]| and ", line) if part.strip(" .;"))
    return items


def _normalized_counter(items: list[str]) -> Counter[str]:
    return Counter(" ".join(str(item).lower().split()) for item in items)


def grade(task: dict, answer: str) -> bool:
    a = (answer or "").strip()
    al = a.lower()
    if "expected_exact" in task:
        return al == str(task["expected_exact"]).strip().lower()
    if "expected_label" in task:
        return al.strip(". ") == str(task["expected_label"]).strip().lower()
    if "expected_entity_set" in task:
        return _normalized_counter(parse_ner_items(a)) == _normalized_counter(task["expected_entity_set"])
    if "expected_contains_all" in task:
        if task.get("category") == "ner":
            return _normalized_counter(parse_ner_items(a)) == _normalized_counter(task["expected_contains_all"])
        return all(str(x).lower() in al for x in task["expected_contains_all"])
    if "expected_contains_any" in task:
        return any(str(x).lower() in al for x in task["expected_contains_any"])
    if "expected_regex" in task:
        return re.search(task["expected_regex"], a) is not None
    if "expected_max_words" in task:
        return len(re.findall(r"\S+", a)) <= int(task["expected_max_words"])
    return False


def run(model_path: str, label: str, files: list[str]) -> None:
    from llama_cpp import Llama

    n_ctx = int(os.getenv("BAKEOFF_N_CTX", "4096"))
    max_tokens = int(os.getenv("BAKEOFF_MAX_TOKENS", "128"))
    t0 = time.time()
    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=2, n_gpu_layers=0, verbose=False)
    print(f"[{label}] load {time.time()-t0:.1f}s", flush=True)
    for path in files:
        cat = path.split("/")[-1].split(".")[0].replace("blind2", "").strip("_") or "mixed"
        tasks = json.load(open(path, encoding="utf-8"))
        raw_ok = acc = acc_ok = 0
        lat = []
        rows = []
        for t in tasks:
            category = t.get("category", cat)
            instr = LOCAL_CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS.get(category, ""))
            s = time.time()
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": LOCAL_MODEL_SYSTEM},
                          {"role": "user", "content": f"{instr}\n{t['prompt']}"}],
                temperature=0.0, max_tokens=max_tokens)
            lat.append(time.time() - s)
            ans = clean_answer((out["choices"][0]["message"].get("content") or ""), category, prompt=t["prompt"])
            correct = grade(t, ans)
            accepted = verify_local_answer(t["prompt"], category, ans)
            raw_ok += correct
            if accepted:
                acc += 1
                acc_ok += correct
                if not correct:
                    print(f"    !! DANGEROUS ACCEPT [{t.get('task_id')}]: {ans[:70]!r}", flush=True)
            rows.append({
                "task_id": t.get("task_id") or t.get("id"),
                "category": category,
                "prompt": t["prompt"],
                "task_spec": t,
                "answer": ans,
                "correct": correct,
                "accepted": accepted,
                "dangerous_wrong_accept": accepted and not correct,
                "latency_seconds": round(lat[-1], 4),
            })
        n = len(tasks)
        summary = {
            "label": label,
            "task_file": path,
            "tasks": n,
            "raw_correct": raw_ok,
            "accepted": acc,
            "accepted_correct": acc_ok,
            "dangerous_wrong_accepts": acc - acc_ok,
            "n_ctx": n_ctx,
            "max_tokens": max_tokens,
            "latency_average_seconds": round(sum(lat) / len(lat), 4),
            "latency_max_seconds": round(max(lat), 4),
        }
        print(f"[{label}] {cat:14} raw {raw_ok}/{n} ({100*raw_ok//n}%) | "
              f"accepted {acc}/{n} accepted-correct {acc_ok}/{acc if acc else 0} | "
              f"lat avg {sum(lat)/len(lat):.1f}s max {max(lat):.1f}s", flush=True)
        output_dir = os.getenv("BAKEOFF_OUTPUT_DIR")
        if output_dir:
            destination = Path(output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            output = destination / f"{label.lower()}_{Path(path).stem}.json"
            output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
            print(f"[{label}] wrote {output}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3:])
