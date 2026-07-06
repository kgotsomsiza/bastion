from __future__ import annotations

import re


def classify_prompt(prompt: str) -> str:
    text = prompt.lower()

    if _contains_any(text, ["debug", "bug", "fix the code", "traceback", "error in this code"]):
        return "code_debugging"

    if _contains_any(text, ["write a function", "implement", "generate code", "return code", "python function"]):
        return "code_generation"

    if _contains_any(text, ["sentiment", "positive", "negative", "neutral"]) and _contains_any(
        text, ["classify", "label", "identify"]
    ):
        return "sentiment"

    if _contains_any(text, ["summarize", "summarise", "summary", "condense"]):
        return "summarization"

    if _contains_any(text, ["named entity", "extract entities", "person", "organization", "location", "date"]):
        return "ner"

    if _contains_any(text, ["logic puzzle", "deduce", "constraint", "which person", "who owns", "must be true"]):
        return "logic"

    if _contains_any(text, ["calculate", "compute", "percentage", "percent", "total cost", "how many"]) or re.search(
        r"\d+\s*[-+*/]\s*\d+", text
    ):
        return "math"

    if _contains_any(text, ["what is", "explain", "define", "how does", "why does"]):
        return "factual"

    return "general"


def _contains_any(text: str, markers: list[str]) -> bool:
    return any(marker in text for marker in markers)
