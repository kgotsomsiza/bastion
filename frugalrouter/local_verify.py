"""Deterministic verification for local-model answers.

The bundled 0.5B model is fast enough for the judge box but not trustworthy
on its own. Every local answer must pass these checks or the router falls
back to the remote path: a wrong local answer risks the accuracy gate, while
a rejected one only costs the tokens we would have spent anyway.
"""
from __future__ import annotations

import re

SENTIMENT_LABELS = {"positive", "negative", "neutral"}

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_META_PHRASES = (
    "as an ai", "i cannot", "i can't", "i'm sorry", "i am sorry",
    "sure,", "certainly", "here is", "here's",
)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "is", "are", "was", "were", "be", "been", "that", "this", "it",
    "its", "as", "by", "from", "will", "has", "have", "had", "their", "they",
}


def verify_local_answer(prompt: str, category: str, answer: str) -> bool:
    text = (answer or "").strip()
    if not text or len(text) > 400:
        return False
    lowered = text.lower()
    if any(phrase in lowered for phrase in _META_PHRASES):
        return False
    if not _explicit_constraints_met(prompt, text):
        return False
    if category == "sentiment":
        return _verify_sentiment(prompt, lowered)
    if category == "ner":
        return _verify_ner(prompt, text)
    if category == "summarization":
        return _verify_summary(prompt, text)
    return False


def _verify_sentiment(prompt: str, lowered_answer: str) -> bool:
    # Only trust the local model on the classic three-label task; YES/NO
    # recommendation questions are subjective and stay remote.
    if not re.search(r"positive|negative|neutral", prompt, flags=re.IGNORECASE):
        return False
    return lowered_answer.strip(". ") in SENTIMENT_LABELS


def _verify_ner(prompt: str, answer: str) -> bool:
    # Every extracted span must literally exist in the source text; a small
    # model inventing or normalizing entities is the failure mode we block.
    # "No entities" claims are unverifiable, so they also go remote.
    if re.search(r"\b(?:none|no [a-z]+ (?:found|present)|n/a)\b", answer, flags=re.IGNORECASE):
        return False
    items: list[str] = []
    for line in answer.splitlines():
        line = re.sub(r"^[\s*•-]*(?:[A-Za-z ]{2,20}:)?", "", line).strip()
        if not line:
            continue
        items.extend(part.strip(" .;") for part in re.split(r"[,;]| and ", line) if part.strip(" .;"))
    if not items:
        return False
    source = prompt.lower()
    return all(item.lower() in source for item in items)


def _verify_summary(prompt: str, answer: str) -> bool:
    # A verbatim run of the source is extraction, not summarization - real
    # summaries compress. Six-plus consecutive source words = reject.
    answer_norm = " ".join(re.findall(r"[a-z]+", answer.lower()))
    prompt_norm = " ".join(re.findall(r"[a-z]+", prompt.lower()))
    answer_tokens = answer_norm.split()
    for start in range(max(1, len(answer_tokens) - 5)):
        if " ".join(answer_tokens[start : start + 6]) in prompt_norm:
            return False

    # Local summaries must stay grounded: most content words should come from
    # the source passage. Abstractive drift goes remote instead of shipping.
    source_words = set(re.findall(r"[a-z]+", prompt.lower()))
    content = [
        w for w in re.findall(r"[a-z]+", answer.lower())
        if len(w) >= 4 and w not in _STOPWORDS
    ]
    if not content:
        return False
    grounded = sum(1 for w in content if w in source_words or w.rstrip("sd") in source_words)
    return grounded / len(content) >= 0.6


def _explicit_constraints_met(prompt: str, answer: str) -> bool:
    words = re.findall(r"[\w'-]+", answer)

    exact = re.search(r"\bexactly\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+words?\b", prompt, flags=re.IGNORECASE)
    if exact:
        want = exact.group(1).lower()
        if len(words) != WORD_NUMBERS.get(want, int(want) if want.isdigit() else -1):
            return False

    if re.search(r"\b(?:in|answer with|reply with|using)\s+one\s+word\b", prompt, flags=re.IGNORECASE) and len(words) != 1:
        return False

    at_most = re.search(
        r"\b(?:no more than|at most|maximum of|under|fewer than|less than)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+words?\b",
        prompt,
        flags=re.IGNORECASE,
    )
    if at_most:
        want = at_most.group(1).lower()
        limit = WORD_NUMBERS.get(want, int(want) if want.isdigit() else 10**6)
        if len(words) > limit:
            return False

    if re.search(r"\b(?:in|as)\s+(?:a\s+)?(?:one|single)\s+sentence\b", prompt, flags=re.IGNORECASE):
        if answer.count("\n") > 0 or len(re.findall(r"[.!?](?:\s|$)", answer.strip())) > 1:
            return False

    forbidden = re.search(r"without\s+(?:using\s+)?the\s+letter\s+'?\"?([a-z])'?\"?", prompt, flags=re.IGNORECASE)
    if forbidden:
        letter = forbidden.group(1).lower()
        if letter in answer.lower():
            return False

    return True
