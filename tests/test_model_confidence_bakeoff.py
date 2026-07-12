import math

import pytest

from scripts.model_confidence_bakeoff import (
    summarize_confidence,
)


def test_summarize_confidence_uses_geometric_mean():
    summary = summarize_confidence([0.25, 1.0], [2.0, 4.0])

    assert summary["first_probability"] == 0.25
    assert summary["min_probability"] == 0.25
    assert summary["mean_probability"] == 0.625
    assert summary["geometric_mean_probability"] == pytest.approx(0.5)
    assert summary["mean_margin"] == 3.0
