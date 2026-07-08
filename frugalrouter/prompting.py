from __future__ import annotations

import re

from frugalrouter.types import Task


# Categories whose tasks are genuinely multi-step (guide: math word problems /
# projections, constraint logic puzzles). These need visible reasoning to be
# correct, so we let the model work step by step and extract the final answer.
REASONING_CATEGORIES = {"math", "logic"}

CATEGORY_INSTRUCTIONS = {
    "factual": "Accurate, concise.",
    "math": "Solve this step by step, showing your work. Then, on the final line, write 'FINAL ANSWER:' followed by only the answer.",
    "sentiment": "Classify sentiment with one word: positive, negative, or neutral. Factual statements with no expressed opinion are neutral.",
    "summarization": "Summarize faithfully; obey format.",
    "ner": "Extract entities; preserve requested labels/format.",
    "code_debugging": "Fix bug; corrected code; minimal explanation.",
    "logic": "Reason through the constraints step by step. Then, on the final line, write 'FINAL ANSWER:' followed by only the answer.",
    "code_generation": "Correct runnable code; code only unless asked.",
    "general": "Answer exactly; be concise.",
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


def clean_answer(text: str, category: str) -> str:
    # Normalize invisible unicode spaces that break exact-match graders.
    answer = text.replace(" ", " ").replace(" ", " ").strip()
    answer = _strip_reasoning_blocks(answer)
    if category in REASONING_CATEGORIES:
        answer = _extract_final_answer(answer)
    for prefix in ("Answer:", "Final answer:", "Final:"):
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix) :].strip()

    answer = _strip_single_code_fence(answer)

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
