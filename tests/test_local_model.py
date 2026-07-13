import math

import numpy as np
import pytest

from frugalrouter.config import load_config
from frugalrouter.providers.local_model import (
    LocalModelProvider,
    _non_thinking_template,
    confidence_from_logits,
    instruction_for_local_task,
)


def test_confidence_from_logits_matches_softmax_and_margin():
    probability, margin = confidence_from_logits(np.array([0.0, 1.0, 3.0]))

    expected = math.exp(3.0) / sum(math.exp(value) for value in (0.0, 1.0, 3.0))
    assert probability == pytest.approx(expected)
    assert margin == pytest.approx(2.0)


def test_local_instruction_is_direct_for_non_explanation_math():
    instruction = instruction_for_local_task("math", "What is 6 * 7? Output only the number.")

    assert "Solve internally" in instruction
    assert "no reasoning" in instruction


def test_local_instruction_keeps_reasoning_request():
    instruction = instruction_for_local_task("logic", "Explain briefly why the answer is YES or NO.")

    assert instruction != instruction_for_local_task("logic", "Answer YES or NO only.")
    assert "Reason carefully" in instruction


def test_non_thinking_template_defines_switch_before_model_template():
    template = _non_thinking_template("{{ messages }}")

    assert template.startswith("{%- set enable_thinking = false -%}")
    assert template.endswith("{{ messages }}")


def test_unconfigured_category_has_impossible_confidence_threshold():
    provider = LocalModelProvider({"local_model": {"confidence_thresholds": {"factual": 0.8}}})

    assert provider.confidence_threshold_for("factual") == 0.8
    assert provider.confidence_threshold_for("logic") > 1.0


def test_v23_config_enables_only_validated_categories():
    config = load_config("config/models.json")
    local = config["local_model"]

    assert set(local["categories"]) == {
        "factual",
        "sentiment",
        "ner",
        "code_debugging",
        "code_generation",
    }
    assert set(local["confidence_thresholds"]) == set(local["categories"])
    assert {"math", "logic", "summarization"}.isdisjoint(local["categories"])
    assert local["disable_thinking"] is True
