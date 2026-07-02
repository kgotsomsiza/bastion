from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from frugalrouter.types import Task


def read_tasks(path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    task_path = Path(path)
    with task_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            tasks.append(
                Task(
                    id=str(payload.get("id", f"task-{line_number}")),
                    input=str(payload["input"]),
                    expected_format=payload.get("expected_format"),
                    metadata={k: v for k, v in payload.items() if k not in {"id", "input", "expected_format"}},
                )
            )
    return tasks


def write_results(path: str | Path, rows: Iterable[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")

