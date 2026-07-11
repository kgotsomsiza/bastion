from frugalrouter.local_verify import verify_local_answer


def test_ner_json_requires_every_acronym():
    prompt = (
        "Extract all acronyms from this text: 'Engineers at NASA and ESA are working on the ISS.' "
        "Format the output as a JSON array of strings."
    )

    assert verify_local_answer(prompt, "ner", '["NASA", "ESA", "ISS"]')
    assert not verify_local_answer(prompt, "ner", '["NASA", "ESA"]')


def test_ner_exact_date_json_preserves_source_spans():
    prompt = (
        "Extract the exact date strings from this sentence: 'We met on 2023-10-01 and close on October 13th.' "
        "Format as a JSON array of strings."
    )

    assert verify_local_answer(prompt, "ner", '["2023-10-01", "October 13th"]')
    assert not verify_local_answer(prompt, "ner", '["2023-10-01", "2023-10-13"]')


def test_ner_pipe_separated_money_is_parsed_and_complete():
    prompt = (
        "Extract the exact monetary values from this text: 'Fees are $15.99, €20.00 and 45 GBP.' "
        "Return them separated by a pipe character '|'."
    )

    assert verify_local_answer(prompt, "ner", "$15.99|€20.00|45 GBP")
    assert not verify_local_answer(prompt, "ner", "$15.99")


def test_ner_space_separated_emails_are_parsed_and_complete():
    prompt = (
        "Extract all email addresses from this text: 'Use test@example.com or admin@site.org.' "
        "Output as a space-separated list."
    )

    assert verify_local_answer(prompt, "ner", "test@example.com admin@site.org")
    assert not verify_local_answer(prompt, "ner", "test@example.com")


def test_ner_abbreviations_survive_apostrophe_inside_quoted_source():
    prompt = (
        "Identify regulations in this memo: 'Follow the European GDPR and California's CCPA.' "
        "List the abbreviations."
    )

    assert verify_local_answer(prompt, "ner", "GDPR, CCPA")
