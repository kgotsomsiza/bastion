from frugalrouter.task_classifier import classify_prompt


def test_date_question_is_not_ner_without_extraction_language():
    assert classify_prompt("What is the date format used in ISO 8601?") == "factual"


def test_entity_extraction_prompt_is_ner():
    assert classify_prompt("Extract people, organizations, locations, and dates from this paragraph.") == "ner"


def test_code_debugging_takes_priority_over_code_generation():
    assert classify_prompt("Debug this Python function and provide corrected code.") == "code_debugging"

