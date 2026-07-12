import math

import numpy as np
import pytest

from scripts.model_confidence_bakeoff import confidence_from_logits, summarize_confidence


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
