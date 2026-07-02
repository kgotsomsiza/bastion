from frugalrouter.config import load_config
from frugalrouter.router import FrugalRouter
from frugalrouter.types import Task


def test_router_uses_local_for_easy_short_answer():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(
        id="test",
        input="Answer with only the capital city: What is the capital of South Africa?",
        expected_format="short_text",
    )

    result = router.run(task)

    assert result.answer.text == "Pretoria"
    assert result.used_remote is False
    assert result.route == "local"


def test_router_marks_risky_when_remote_disabled():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(
        id="test",
        input="Write a function that computes a rolling median from a stream.",
        expected_format="short_text",
    )

    result = router.run(task)

    assert result.used_remote is False
    assert result.route == "local_remote_disabled"
    assert result.fallback_reason is not None

