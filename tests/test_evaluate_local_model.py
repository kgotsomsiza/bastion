import json

from scripts.evaluate_local_model import _grade, _load_rows


def test_holdout_jsonl_is_converted_to_strict_specs(tmp_path):
    source = tmp_path / "holdout.jsonl"
    row = {
        "family": "ner_B_holdout",
        "category": "ner",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "instruction\nFind names: Ada met Linus."},
            {"role": "assistant", "content": "Ada, Linus"},
        ],
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert _load_rows(source) == [
        {
            "task_id": "ner_B_holdout-0001",
            "category": "ner",
            "prompt": "Find names: Ada met Linus.",
            "expected_entity_set": ["Ada", "Linus"],
        }
    ]


def test_ner_grade_rejects_extra_source_entities():
    spec = {"expected_entity_set": ["Ada", "Linus"]}
    assert _grade("Linus, Ada", spec)["passed"] is True
    assert _grade("Ada, Linus, Grace", spec)["passed"] is False
