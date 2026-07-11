"""Deterministic verification for local-model answers.

The bundled 0.5B model is fast enough for the judge box but not trustworthy
on its own. Every local answer must pass these checks or the router falls
back to the remote path: a wrong local answer risks the accuracy gate, while
a rejected one only costs the tokens we would have spent anyway.
"""
from __future__ import annotations

import json
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
    # Contrast or negation in the text signals nuance (mixed reviews, double
    # negatives like "wasn't the worst... but") - measured to trip the local
    # model, so those stay remote. Straightforward emotional language only.
    if re.search(r"\b(?:but|however|although|though|yet|not|never|hardly|barely)\b|n't\b", prompt, flags=re.IGNORECASE):
        return False
    return lowered_answer.strip(". ") in SENTIMENT_LABELS


def _verify_ner(prompt: str, answer: str) -> bool:
    if re.search(r"\b(?:none|no [a-z]+ (?:found|present)|n/a)\b", answer, flags=re.IGNORECASE):
        return False
    items = _parse_ner_items(prompt, answer)
    if not items:
        return False

    source_text = _ner_source_text(prompt)
    source = source_text.lower()
    for item in items:
        pattern = re.escape(item.lower())
        if not re.search(rf"(?<![\w$€£#.]){pattern}(?![\w%])", source):
            return False

    required = _required_ner_items(prompt, source_text)
    if required:
        normalized_items = {_normalize_ner_item(item) for item in items}
        normalized_required = {_normalize_ner_item(item) for item in required}
        if normalized_items != normalized_required:
            return False
    return True


def _parse_ner_items(prompt: str, answer: str) -> list[str]:
    stripped = answer.strip()
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list) or not payload or not all(isinstance(item, str) for item in payload):
            return []
        return [item.strip() for item in payload if item.strip()]

    if "|" in stripped and re.search(r"\bpipe(?:-separated| character)?\b", prompt, flags=re.IGNORECASE):
        return [item.strip() for item in stripped.split("|") if item.strip()]

    if re.search(r"\bemail addresses?\b", prompt, flags=re.IGNORECASE):
        emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", stripped)
        if emails:
            return emails

    items: list[str] = []
    for line in stripped.splitlines():
        line = re.sub(r"^[\s*•-]*(?:[A-Za-z ]{2,20}:)?", "", line).strip()
        if not line:
            continue
        items.extend(part.strip(" .;") for part in re.split(r"[,;]| and ", line) if part.strip(" .;"))
    return items


def _ner_source_text(prompt: str) -> str:
    marked = re.search(
        r":\s*'(.*?)'\s*(?=(?:List|Return|Output|Format)\b|$)",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if marked:
        return marked.group(1)
    quoted = re.findall(r"'([^'\n]{10,})'", prompt)
    return max(quoted, key=len) if quoted else prompt


def _required_ner_items(prompt: str, source: str) -> list[str]:
    lowered = prompt.lower()
    if "acronym" in lowered or "abbreviation" in lowered:
        return re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", source)
    if "email address" in lowered:
        return re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", source)
    if "monetary value" in lowered:
        return re.findall(
            r"(?:[$€£]\s?\d+(?:,\d{3})*(?:\.\d+)?|"
            r"\b(?:USD|EUR|GBP|ZAR|R)\s?\d+(?:,\d{3})*(?:\.\d+)?|"
            r"\b\d+(?:,\d{3})*(?:\.\d+)?\s?(?:USD|EUR|GBP|ZAR))\b",
            source,
        )
    if re.search(r"\b(?:all|exact)\s+date", lowered):
        months = (
            r"January|February|March|April|May|June|July|August|September|October|November|December"
        )
        ordinals = (
            r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|"
            r"thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|"
            r"twentieth|twenty-first|twenty-second|twenty-third|twenty-fourth|twenty-fifth|"
            r"twenty-sixth|twenty-seventh|twenty-eighth|twenty-ninth|thirtieth|thirty-first"
        )
        return re.findall(
            rf"\b(?:\d{{4}}-\d{{2}}-\d{{2}}|(?:{months})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?|"
            rf"(?:{ordinals})\s+of\s+(?:{months})(?:,\s*\d{{4}})?)\b",
            source,
            flags=re.IGNORECASE,
        )
    return []


def _normalize_ner_item(item: str) -> str:
    return re.sub(r"\s+", " ", item.strip()).lower()


def _verify_ner_legacy(prompt: str, answer: str) -> bool:
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
    # Clean-boundary match: a bare "50" must not pass because "$50" appears in
    # the source - the surrounding characters must not be part of the span.
    import re as _re
    for item in items:
        pattern = _re.escape(item.lower())
        if not _re.search(rf"(?<![\w$€£#.]){pattern}(?![\w%])", source):
            return False
    return True


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
