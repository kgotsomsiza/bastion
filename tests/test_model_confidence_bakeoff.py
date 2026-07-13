import math

import pytest

from scripts.model_confidence_bakeoff import (
    checkpoint_path_for,
    grade_task,
    load_checkpoint_rows,
    parse_ner_items,
    summarize_confidence,
)


def test_summarize_confidence_uses_geometric_mean():
    summary = summarize_confidence([0.25, 1.0], [2.0, 4.0])

    assert summary["first_probability"] == 0.25
    assert summary["min_probability"] == 0.25
    assert summary["mean_probability"] == 0.625
    assert summary["geometric_mean_probability"] == pytest.approx(0.5)
    assert summary["mean_margin"] == 3.0


def test_checkpoint_rows_round_trip(tmp_path):
    output = tmp_path / "report.json"
    checkpoint = checkpoint_path_for(output)
    checkpoint.write_text('{"task_id":"one","correct":true}\n', encoding="utf-8")

    assert checkpoint.name == "report.json.rows.jsonl"
    assert load_checkpoint_rows(checkpoint) == [{"task_id": "one", "correct": True}]


def test_ner_grade_rejects_extra_entities():
    spec = {
        "category": "ner",
        "expected_contains_all": ["Alice", "Bob"],
    }

    assert grade_task("Alice, Bob", spec)["passed"] is True
    assert grade_task("Alice, Bob, Mallory", spec)["passed"] is False


def test_ner_parser_supports_json_lists_and_pipe_separators():
    assert parse_ner_items('["NASA", "ESA"]') == ["NASA", "ESA"]
    assert parse_ner_items("NASA | ESA") == ["NASA", "ESA"]
