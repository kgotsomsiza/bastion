from scripts.gen_finetune_data_v2 import clean_sentiment_candidate, parse_string_array


def test_parse_string_array_accepts_plain_and_fenced_json():
    assert parse_string_array('["one", "two"]') == ["one", "two"]
    assert parse_string_array('```json\n["one", "two"]\n```') == ["one", "two"]


def test_parse_string_array_rejects_non_string_values():
    assert parse_string_array('["one", 2]') == []
    assert parse_string_array("not json") == []


def test_clean_sentiment_candidate_rejects_generation_instructions():
    assert clean_sentiment_candidate("One per line") is None
    assert clean_sentiment_candidate("1. No numbering") is None
    assert clean_sentiment_candidate("Each must be clearly dissatisfied") is None


def test_clean_sentiment_candidate_normalizes_numbering():
    assert clean_sentiment_candidate("1. The service was genuinely excellent today.") == (
        "The service was genuinely excellent today."
    )
