import math

import numpy as np
import pytest

from scripts.model_confidence_bakeoff import (
    confidence_from_logits,
    instruction_for_task,
    non_thinking_template,
    summarize_confidence,
)


def test_non_thinking_template_defines_switch_before_model_template():
    template = non_thinking_template("{{ messages }}")

    assert template.startswith("{%- set enable_thinking = false -%}")
    assert template.endswith("{{ messages }}")


def test_reasoning_task_without_explanation_gets_direct_local_instruction():
    instruction = instruction_for_task("math", "What is 8 * 7? Output only the number.")

    assert "Solve internally" in instruction
    assert "no reasoning" in instruction


def test_reasoning_task_with_explanation_keeps_category_instruction():
    instruction = instruction_for_task("logic", "Explain briefly why the answer is YES or NO.")

    assert instruction != instruction_for_task("logic", "Answer YES or NO only.")
    assert "Reason carefully" in instruction


def test_confidence_from_logits_matches_softmax_and_margin():
    probability, margin = confidence_from_logits(np.array([0.0, 1.0, 3.0]))

    expected = math.exp(3.0) / sum(math.exp(value) for value in (0.0, 1.0, 3.0))
    assert probability == pytest.approx(expected)
    assert margin == pytest.approx(2.0)


def test_summarize_confidence_uses_geometric_mean():
    summary = summarize_confidence([0.25, 1.0], [2.0, 4.0])

    assert summary["first_probability"] == 0.25
    assert summary["min_probability"] == 0.25
    assert summary["mean_probability"] == 0.625
    assert summary["geometric_mean_probability"] == pytest.approx(0.5)
    assert summary["mean_margin"] == 3.0
