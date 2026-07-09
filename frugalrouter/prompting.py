from __future__ import annotations

import re

from frugalrouter.types import Task


# Categories whose tasks are genuinely multi-step (guide: math word problems /
# projections, constraint logic puzzles). These need visible reasoning to be
# correct, so we let the model work step by step and extract the final answer.
REASONING_CATEGORIES = {"math", "logic"}

CATEGORY_INSTRUCTIONS = {
    "factual": "Answer accurately and concisely. Follow the requested format.",
    "math": "Solve carefully. If the prompt asks for reasoning, include concise work. Put the final result on the last line as 'FINAL ANSWER:' followed by only the answer.",
    "sentiment": "Classify the sentiment toward the target being asked about, not isolated words. Handle negation and mixed phrasing carefully. Matching expectations in a factual update is neutral. Follow the requested format.",
    "summarization": "Summarize faithfully; obey the requested format and length.",
    "ner": "Extract only the requested entities. Preserve the exact source text spans; do not normalize dates, names, or values unless asked.",
    "code_debugging": "Fix the bug. Output corrected code unless the prompt asks for explanation.",
    "logic": "Reason carefully. If the prompt asks for reasoning, include concise work. Put the final result on the last line as 'FINAL ANSWER:' followed by only the answer.",
    "code_generation": "Output correct runnable code only unless the prompt asks for explanation.",
    "general": "Answer exactly and concisely. Follow the requested format.",
}


def user_prompt(task: Task, category: str, no_reasoning: bool = False) -> str:
    format_hint = f"\nFormat: {task.expected_format}" if task.expected_format else ""
    instruction = CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS["general"])
    if category in REASONING_CATEGORIES:
        # Let the model reason freely; no terse "Answer:" cue or no-reasoning
        # directive that would suppress the step-by-step working.
        return f"{instruction}\n{task.input}{format_hint}"
    directive = "\nDo not show reasoning or thoughts. Output only the final answer." if no_reasoning else ""
    return f"{instruction}\n{task.input}{format_hint}{directive}\nAnswer:"


def _extract_final_answer(text: str) -> str:
    """Pull the answer after the last 'FINAL ANSWER:' marker.

    Reasoning categories work step by step and end with 'FINAL ANSWER: X';
    we submit just X so the output is clean for any grader. Falls back to the
    text as-is if the marker is absent (e.g. the model answered directly).
    """
    markers = list(re.finditer(r"\**\s*final\s+answer\s*\**\s*:?\s*", text, flags=re.IGNORECASE))
    if not markers:
        return text
    tail = text[markers[-1].end() :].strip()
    first_line = tail.split("\n", 1)[0].strip().strip("*").strip()
    return first_line or tail


def looks_like_reasoning_spill(text: str) -> bool:
    """Detect thinking-mode deliberation leaked into the answer text.

    Gemma-style spill starts with a bare "thought" line; other models leak
    meta-planning phrases like "We need answer user".
    """
    stripped = text.lstrip()
    if re.match(r"^thought\s*\n", stripped, flags=re.IGNORECASE):
        return True
    return bool(re.match(r"^(?:we need to|we need answer|the user wants|let me think)", stripped, flags=re.IGNORECASE))


def clean_answer(text: str, category: str, prompt: str | None = None) -> str:
    # Normalize invisible unicode spaces that break exact-match graders.
    answer = text.replace(" ", " ").replace(" ", " ").strip()
    answer = _strip_reasoning_blocks(answer)
    if category in REASONING_CATEGORIES and not prompt_wants_explanation(prompt):
        answer = _extract_final_answer(answer)
    for prefix in ("Answer:", "Final answer:", "Final:"):
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix) :].strip()

    answer = _strip_single_code_fence(answer)
    answer = _extract_leading_inline_code_when_exact_requested(answer, prompt)
    answer = _strip_inline_code_ticks(answer)
    answer = _strip_bare_call_when_identifier_requested(answer, prompt)
    answer = _format_time_when_hhmm_requested(answer, prompt)

    if category == "sentiment" and not prompt_wants_explanation(prompt) and not prompt_wants_structured_answer(prompt):
        label = _extract_sentiment_label(answer)
        if label:
            return label

    if category not in {"code_generation", "code_debugging"}:
        answer = _strip_outer_quotes(answer)

    return answer


def _strip_reasoning_blocks(text: str) -> str:
    """Drop <think>...</think> deliberation that reasoning models emit.

    An unclosed <think> means the model was truncated mid-reasoning; nothing
    after the tag is answer material, so cut there (often leaving an empty
    answer, which the router's truncation rescue then retries).
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.split(r"<think>", cleaned, flags=re.IGNORECASE)[0]
    return cleaned.strip()


def _strip_single_code_fence(text: str) -> str:
    """Unwrap an answer that is exactly one fenced code block.

    Models fence code even when told "code only"; execution- or
    exact-match-based graders will not accept the fence markers.
    """
    match = re.match(r"^```[a-zA-Z0-9_+-]*\r?\n(.*?)\r?\n?```$", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _strip_outer_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def _strip_inline_code_ticks(text: str) -> str:
    match = re.match(r"^`([^`\r\n]+)`$", text.strip())
    if match:
        return match.group(1).strip()
    return text


def _extract_leading_inline_code_when_exact_requested(text: str, prompt: str | None) -> str:
    if not prompt or not re.search(
        r"\b(?:exact|exactly|only|just|nothing else|no extra text|give just|output just)\b",
        prompt,
        flags=re.IGNORECASE,
    ):
        return text
    match = re.match(r"^`([^`\r\n]+)`(?:\s|$)", text.strip())
    return match.group(1).strip() if match else text


def _strip_bare_call_when_identifier_requested(text: str, prompt: str | None) -> str:
    if not prompt:
        return text
    if not re.search(
        r"\b(?:method name|keyword|property(?: name)?|tag name|command name|clause|one word|word only)\b",
        prompt,
        flags=re.IGNORECASE,
    ):
        return text
    match = re.fullmatch(r"([A-Za-z_][\w.-]*)\(\)", text.strip())
    return match.group(1) if match else text


def _format_time_when_hhmm_requested(text: str, prompt: str | None) -> str:
    if not prompt or not re.search(r"\bHH:MM\b", prompt, flags=re.IGNORECASE):
        return text
    match = re.fullmatch(r"(\d{1,2}):(\d{2})\s*([AP]M)", text.strip(), flags=re.IGNORECASE)
    if not match:
        return text
    hour, minute, suffix = match.groups()
    return f"{int(hour):02d}:{minute} {suffix.upper()}"


def _extract_sentiment_label(text: str) -> str | None:
    stripped = text.strip()
    patterns = [
        r"^(positive|negative|neutral)\.?$",
        r"^(positive|negative|neutral)\s*:",
        r"^(?:the\s+)?sentiment\s+(?:is|:)\s*(positive|negative|neutral)\.?$",
        r"^(?:answer|label)\s*:\s*(positive|negative|neutral)\.?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return None


def prompt_wants_explanation(prompt: str | None) -> bool:
    if not prompt:
        return False
    return bool(
        re.search(
            r"\b(?:explain|why|justify|justification|reason(?:ing|s)?|rationale|because|show\s+(?:your\s+)?work)\b",
            prompt,
            flags=re.IGNORECASE,
        )
    )


def prompt_wants_structured_answer(prompt: str | None) -> bool:
    if not prompt:
        return False
    return bool(
        re.search(
            r"\b(?:json|object|array|mapping|map|dictionary|valid\s+json|review\s+number|reviews?)\b",
            prompt,
            flags=re.IGNORECASE,
        )
    )
