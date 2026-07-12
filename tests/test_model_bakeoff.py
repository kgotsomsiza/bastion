from scripts.model_bakeoff import grade, parse_ner_items


def test_parse_ner_items_handles_lines_and_commas():
    assert parse_ner_items("Names: Ada, Linus\nGrace") == ["Ada", "Linus", "Grace"]


def test_ner_grade_requires_exact_entity_set():
    task = {"category": "ner", "expected_entity_set": ["Ada", "Linus"]}
    assert grade(task, "Linus, Ada") is True
    assert grade(task, "Ada, Linus, Grace") is False
    assert grade(task, "Ada") is False


def test_legacy_ner_contains_all_is_also_strict():
    task = {"category": "ner", "expected_contains_all": ["Ada", "Linus"]}
    assert grade(task, "Ada, Linus") is True
    assert grade(task, "Ada, Linus, Grace") is False
