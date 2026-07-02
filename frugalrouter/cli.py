from __future__ import annotations

import argparse
from dataclasses import asdict

from frugalrouter.config import env_flag, load_config
from frugalrouter.decision_log import DecisionLogger
from frugalrouter.io import read_tasks, write_results
from frugalrouter.router import FrugalRouter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FrugalRouter on a JSONL task file.")
    parser.add_argument("--tasks", default="data/sample_tasks.jsonl", help="Input JSONL task file.")
    parser.add_argument("--output", default="outputs/results.jsonl", help="Output JSONL result file.")
    parser.add_argument("--config", default="config/models.json", help="Router config JSON.")
    parser.add_argument("--allow-remote", action="store_true", help="Allow Fireworks API fallback.")
    parser.add_argument("--decision-log", default=None, help="Decision log JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    allow_remote = args.allow_remote or env_flag("FRUGAL_ALLOW_REMOTE", default=False)
    router = FrugalRouter(config=config, allow_remote=allow_remote)
    logger = DecisionLogger(args.decision_log)

    rows = []
    for task in read_tasks(args.tasks):
        result = router.run(task)
        logger.write(task, result)
        rows.append(
            {
                "id": result.task_id,
                "answer": result.answer.text,
                "route": result.route,
                "confidence": round(result.verification.confidence, 4),
                "used_remote": result.used_remote,
                "fallback_reason": result.fallback_reason,
                "usage": {
                    "prompt_tokens": result.answer.prompt_tokens,
                    "completion_tokens": result.answer.completion_tokens,
                    "latency_ms": result.answer.latency_ms,
                    "model": result.answer.model,
                    "provider": result.answer.provider,
                },
                "debug": asdict(result.verification),
            }
        )

    write_results(args.output, rows)
    remote_count = sum(1 for row in rows if row["used_remote"])
    print(f"Wrote {len(rows)} results to {args.output}. Remote calls: {remote_count}.")


if __name__ == "__main__":
    main()

