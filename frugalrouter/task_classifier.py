from __future__ import annotations

import re


def classify_prompt(prompt: str) -> str:
    text = prompt.lower()

    if _contains_any(
        text,
        [
            "debug",
            "bug",
            "fix the code",
            "traceback",
            "error in this code",
            "corrected code",
            "syntax error",
            "typeerror",
            "does not crash",
            "fix this",
        ],
    ):
        return "code_debugging"

    if _contains_any(
        text,
        [
            "write a function",
            "implement",
            "generate code",
            "return code",
            "python function",
            "javascript function",
            "sql query",
            "write a sql",
            "line of python",
            "code only",
        ],
    ):
        return "code_generation"

    if _contains_any(text, ["sentiment", "positive", "negative", "neutral"]) and _contains_any(
        text, ["classify", "label", "identify"]
    ):
        return "sentiment"

    if _contains_any(text, ["summarize", "summarise", "summary", "condense", "headline", "main point", "tl;dr", "tldr"]):
        return "summarization"

    if _contains_any(text, ["named entity", "extract entities"]) or (
        _contains_any(text, ["extract", "identify", "list", "find"])
        and _contains_any(
            text,
            ["person", "people", "organization", "organisation", "location", "date", "entities", "names", "product"],
        )
    ):
        return "ner"

    if _contains_any(
        text,
        [
            "logic puzzle",
            "deduce",
            "constraint",
            "which person",
            "who owns",
            "must be true",
            "cannot both",
            "exactly one",
            "older than",
            "younger than",
            "taller than",
            "shorter than",
            "comes next",
            "sequence",
            "all but",
            "sitting between",
            "directly between",
            "to the left of",
            "to the right of",
            "if yesterday",
            "if tomorrow",
            "if all",
            "if some",
            "if no ",
        ],
    ):
        return "logic"

    if _contains_any(
        text,
        ["calculate", "compute", "percentage", "percent", "total cost", "how many", "how far", "how much", "% of"],
    ) or re.search(r"\d+\s*[-+*/]\s*\d+", text):
        return "math"

    if _contains_any(
        text,
        [
            "what is",
            "explain",
            "define",
            "how does",
            "why does",
            "what year",
            "what does",
            "which company",
            "which country",
            "capital of",
            "stand for",
            "chemical symbol",
        ],
    ):
        return "factual"

    return "general"


def _contains_any(text: str, markers: list[str]) -> bool:
    return any(marker in text for marker in markers)
