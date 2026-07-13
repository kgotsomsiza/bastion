from frugalrouter.local_verify import verify_confident_local_answer, verify_local_answer


def test_original_local_verifier_still_rejects_unverifiable_factual_answer():
    assert verify_local_answer("What is Canada's capital?", "factual", "Ottawa") is False


def test_confident_verifier_allows_factual_answer_that_meets_constraints():
    assert verify_confident_local_answer(
        "What is Canada's capital? Output only the city name.",
        "factual",
        "Ottawa",
    ) is True


def test_confident_verifier_rejects_explicit_word_count_violation():
    assert verify_confident_local_answer(
        "Answer in exactly two words.",
        "factual",
        "one word too many",
    ) is False


def test_confident_verifier_keeps_nuanced_sentiment_remote():
    assert verify_confident_local_answer(
        "Classify as positive, negative, or neutral: It was good, but not worth the price.",
        "sentiment",
        "negative",
    ) is False


def test_confident_verifier_rejects_invented_ner_span():
    assert verify_confident_local_answer(
        "Extract every person name: Alice went home.",
        "ner",
        "Robert",
    ) is False
