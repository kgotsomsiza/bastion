from scripts.evaluate_structured_ner import evaluate


def test_evaluate_structured_ner_counts_only_exact_local_answers():
    tasks = [
        {
            "task_id": "email",
            "prompt": "Extract all email addresses from: Use a@example.com or b@example.org.",
            "expected_contains_all": ["a@example.com", "b@example.org"],
        },
        {
            "task_id": "mixed",
            "prompt": "Extract monetary values and dates from: Pay $8 on July 2.",
            "expected_contains_all": ["$8", "July 2"],
        },
    ]

    summary, rows = evaluate(tasks)

    assert summary == {
        "tasks": 2,
        "accepted": 1,
        "accepted_correct": 1,
        "dangerous_accepts": 0,
        "coverage": 0.5,
    }
    assert rows[0]["correct"] is True
    assert rows[1]["accepted"] is False
