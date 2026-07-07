from frugalrouter.prompting import clean_answer


def test_strips_single_code_fence():
    fenced = "```python\ndef add_one(x):\n    return x + 1\n```"
    assert clean_answer(fenced, "code_generation") == "def add_one(x):\n    return x + 1"


def test_strips_fence_without_language_tag():
    assert clean_answer("```\nSELECT 1;\n```", "code_generation") == "SELECT 1;"


def test_keeps_fences_when_answer_has_surrounding_text():
    mixed = "Here is the fix:\n```python\nx = 1\n```"
    assert clean_answer(mixed, "code_debugging") == mixed


def test_normalizes_unicode_spaces():
    assert clean_answer("reduced memory by 30 %", "summarization") == "reduced memory by 30 %"


def test_strips_answer_prefix_and_quotes():
    assert clean_answer("Answer: 'Canberra'", "factual") == "Canberra"
