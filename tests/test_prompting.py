from frugalrouter.prompting import clean_answer, looks_like_reasoning_spill, user_prompt
from frugalrouter.types import Task


def test_detects_gemma_thought_spill():
    assert looks_like_reasoning_spill("thought\n*   The user wants X.")
    assert looks_like_reasoning_spill("We need answer user. They want a fix.")
    assert not looks_like_reasoning_spill("Canberra")
    assert not looks_like_reasoning_spill("def add_one(x):\n    return x + 1")


def test_no_reasoning_directive_appended():
    # factual is a non-reasoning category, where the directive applies.
    task = Task(id="t", input="What is the capital of France?")
    assert "Do not show reasoning" in user_prompt(task, "factual", no_reasoning=True)
    assert "Do not show reasoning" not in user_prompt(task, "factual")


def test_reasoning_category_prompts_for_step_by_step():
    task = Task(id="t", input="A train travels 120 km in 2 hours; how long for 300 km?")
    # math/logic ask for working + a FINAL ANSWER marker, and ignore no_reasoning.
    for cat in ("math", "logic"):
        prompt = user_prompt(task, cat, no_reasoning=True)
        assert "FINAL ANSWER" in prompt
        assert "Do not show reasoning" not in prompt


def test_reasoning_category_extracts_final_answer():
    from frugalrouter.prompting import clean_answer

    reasoned = "Step 1: 120/2 = 60 km/h.\nStep 2: 300/60 = 5 h.\n**FINAL ANSWER:** 5 hours"
    assert clean_answer(reasoned, "math") == "5 hours"


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


def test_strips_closed_think_block():
    text = "<think>Let me reason about this.</think>\nCanberra"
    assert clean_answer(text, "factual") == "Canberra"


def test_unclosed_think_block_truncates_to_empty():
    text = "<think>We need answer user. They want fix bug so function"
    assert clean_answer(text, "code_debugging") == ""
