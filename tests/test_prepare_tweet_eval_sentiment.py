from scripts.prepare_tweet_eval_sentiment import _eligible_rows, _select_balanced


def test_eligible_rows_deduplicates_and_applies_runtime_gate():
    dataset = [
        {"text": "Great service today", "label": 2},
        {"text": "Great   service today", "label": 2},
        {"text": "It was not good", "label": 0},
        {"text": "The package arrived Tuesday", "label": 1},
    ]
    rows = _eligible_rows(dataset, ["negative", "neutral", "positive"])
    assert [(row["text"], row["label"]) for row in rows] == [
        ("Great service today", "positive"),
        ("The package arrived Tuesday", "neutral"),
    ]


def test_select_balanced_is_balanced_and_reproducible():
    rows = [
        {"label": label, "text": f"{label}-{index}"}
        for label in ("negative", "neutral", "positive")
        for index in range(5)
    ]
    first = _select_balanced(rows, per_label=3, seed=7)
    second = _select_balanced(rows, per_label=3, seed=7)
    assert first == second
    assert {label: sum(row["label"] == label for row in first) for label in ("negative", "neutral", "positive")} == {
        "negative": 3,
        "neutral": 3,
        "positive": 3,
    }
