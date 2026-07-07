from __future__ import annotations

import ast
import operator
import re


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def solve_simple_math(prompt: str) -> tuple[str, str] | None:
    text = prompt.strip()
    lower = text.lower()

    percent = _solve_percentage(lower)
    if percent is not None:
        return percent, "percentage_pattern"

    if not _looks_like_direct_math_request(lower):
        return None

    expression = _extract_expression(text)
    if not expression:
        return None

    try:
        value = _safe_eval(expression)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None

    return _format_number(value), "direct_arithmetic_expression"


def _looks_like_direct_math_request(text: str) -> bool:
    markers = [
        "calculate",
        "compute",
        "evaluate",
        "solve",
        "what is",
        "return only the number",
        "give only the number",
    ]
    return any(marker in text for marker in markers)


def _extract_expression(prompt: str) -> str | None:
    compact = prompt.replace("×", "*").replace("÷", "/")
    candidates = re.findall(r"[-+*/().\d\s]{3,}", compact)
    candidates = [candidate.strip() for candidate in candidates if re.search(r"\d", candidate)]
    candidates = [candidate for candidate in candidates if re.search(r"[-+*/]", candidate)]
    if not candidates:
        return None
    return max(candidates, key=len)


def _solve_percentage(text: str) -> str | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(-?\d+(?:\.\d+)?)", text)
    if match:
        return _format_number(float(match.group(1)) * float(match.group(2)) / 100)

    match = re.search(r"(-?\d+(?:\.\d+)?)\s+is\s+what\s+(?:%|percent)\s+of\s+(-?\d+(?:\.\d+)?)", text)
    if match and float(match.group(2)) != 0:
        return _format_number(float(match.group(1)) * 100 / float(match.group(2)))

    match = re.search(
        r"(?:increase|raise)\s+(-?\d+(?:\.\d+)?)\s+by\s+(-?\d+(?:\.\d+)?)\s*(?:%|percent)",
        text,
    )
    if match:
        base = float(match.group(1))
        pct = float(match.group(2))
        return _format_number(base * (1 + pct / 100))

    match = re.search(
        r"(?:decrease|reduce|discount)\s+(-?\d+(?:\.\d+)?)\s+by\s+(-?\d+(?:\.\d+)?)\s*(?:%|percent)",
        text,
    )
    if match:
        base = float(match.group(1))
        pct = float(match.group(2))
        return _format_number(base * (1 - pct / 100))

    return None


def _safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return float(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.10g}"

