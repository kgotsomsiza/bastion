from __future__ import annotations

from frugalrouter.types import Task


CATEGORY_INSTRUCTIONS = {
    "factual": "Accurate, concise.",
    "math": "Solve; final answer first.",
    "sentiment": "Classify sentiment with one word: positive, negative, or neutral. Factual statements with no expressed opinion are neutral.",
    "summarization": "Summarize faithfully; obey format.",
    "ner": "Extract entities; preserve requested labels/format.",
    "code_debugging": "Fix bug; corrected code; minimal explanation.",
    "logic": "Solve constraints; concise answer.",
    "code_generation": "Correct runnable code; code only unless asked.",
    "general": "Answer exactly; be concise.",
}


def user_prompt(task: Task, category: str) -> str:
    format_hint = f"\nFormat: {task.expected_format}" if task.expected_format else ""
    instruction = CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS["general"])
    return f"{instruction}\n{task.input}{format_hint}\nAnswer:"


def clean_answer(text: str, category: str) -> str:
    answer = text.strip()
    for prefix in ("Answer:", "Final answer:", "Final:"):
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix) :].strip()

    if category not in {"code_generation", "code_debugging"}:
        answer = _strip_outer_quotes(answer)

    return answer


def _strip_outer_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text
