from frugalrouter.math_solver import solve_simple_math


def test_solves_parenthesized_expression():
    assert solve_simple_math("Calculate (8 + 4) * 3. Return only the number.") == (
        "36",
        "direct_arithmetic_expression",
    )


def test_solves_percentage_of_pattern():
    assert solve_simple_math("What is 12.5% of 80? Return only the number.") == ("10", "percentage_pattern")


def test_ignores_non_math_prompt_with_numbers():
    assert solve_simple_math("Explain what happened in 2024 and 2025.") is None

