from scripts.prepare_conll_ner import _detokenize, _extract_spans


def test_detokenize_handles_punctuation_and_possessives():
    assert _detokenize(["Alice", "'s", "team", ",", "Acme", "."]) == "Alice's team, Acme."


def test_extract_spans_returns_contiguous_entities():
    names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    tokens = ["Ada", "Lovelace", "joined", "Open", "AI", "in", "London", "."]
    tags = [1, 2, 0, 3, 4, 0, 5, 0]
    assert _extract_spans(tokens, tags, names, "PER") == ["Ada Lovelace"]
    assert _extract_spans(tokens, tags, names, "ORG") == ["Open AI"]
    assert _extract_spans(tokens, tags, names, "LOC") == ["London"]
