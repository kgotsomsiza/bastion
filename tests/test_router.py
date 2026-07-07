import json

from frugalrouter.config import load_config
from frugalrouter.io import read_tasks, write_results
from frugalrouter.model_policy import ModelPolicy
from frugalrouter.providers.fireworks import FireworksError
from frugalrouter.router import FrugalRouter
from frugalrouter.types import Answer, Task


def test_router_uses_local_for_direct_arithmetic():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(id="test", input="Calculate 42 * 17. Return only the number.")

    result = router.run(task)

    assert result.answer.text == "714"
    assert result.used_remote is False
    assert result.route == "local"
    assert result.category == "math"


def test_router_uses_local_for_direct_percentage():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(id="test", input="What is 25 percent of 200? Return only the number.")

    result = router.run(task)

    assert result.answer.text == "50"
    assert result.used_remote is False
    assert result.route == "local"


def test_router_uses_local_for_exact_response_instruction():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(id="test", input="Reply with exactly: OK")

    result = router.run(task)

    assert result.answer.text == "OK"
    assert result.used_remote is False
    assert result.route == "local"


def test_router_does_not_shortcut_negated_sentiment():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(id="test", input="Classify the sentiment as positive, neutral, or negative: The docs are not good.")

    result = router.run(task)

    assert result.route == "local_remote_disabled"
    assert result.used_remote is False


def test_router_does_not_shortcut_exact_response_with_alternatives():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(id="test", input="Answer with exactly 'yes' or 'no': Is the Atlantic larger than the Pacific?")

    result = router.run(task)

    assert result.route == "local_remote_disabled"
    assert result.answer.text == ""


def test_router_does_not_use_exact_response_for_length_constraint():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(id="test", input="Answer with exactly three words: Explain ROCm.")

    result = router.run(task)

    assert result.answer.text == ""
    assert result.route == "local_remote_disabled"


def test_router_uses_remote_for_general_tasks_without_local_shortcut():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=True)

    class FakeProvider:
        def answer(self, task, model=None, category="general"):
            return Answer(text=f"{category}:{model}:ok", provider="fireworks", model=model or "fake")

    router.remote = FakeProvider()
    task = Task(id="test", input="Explain in one sentence what ROCm is.")

    result = router.run(task)

    assert result.route == "remote"
    assert result.used_remote is True
    assert result.category == "factual"
    assert "gemma" in result.answer.text


def test_router_falls_back_to_empty_local_when_remote_fails():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=True)

    class FailingProvider:
        def answer(self, task, model=None, category="general"):
            raise RuntimeError("simulated remote outage")

    router.remote = FailingProvider()
    task = Task(id="test", input="Write a function that computes a rolling median from a stream.")

    result = router.run(task)

    assert result.route == "remote_error"
    assert result.used_remote is False
    assert "simulated remote outage" in result.fallback_reason


def test_router_tries_next_model_after_404():
    config = load_config("config/models.json")
    config["allowed_models"] = ["missing-model", "accounts/fireworks/models/minimax-m3"]
    config["model_policy"]["factual"] = ["missing-model", "minimax-m3"]
    router = FrugalRouter(config=config, allow_remote=True)

    class FlakyProvider:
        def answer(self, task, model=None, category="general"):
            if "missing-model" in model:
                raise FireworksError("not found", status_code=404)
            return Answer(text=f"{model}:ok", provider="fireworks", model=model or "fake")

    router.remote = FlakyProvider()

    result = router.run(Task(id="test", input="Explain what ROCm is."))

    assert result.route == "remote"
    assert result.used_remote is True
    assert result.answer.model == "accounts/fireworks/models/minimax-m3"


def test_router_retries_404_with_prefixed_model_name():
    config = load_config("config/models.json")
    config["allowed_models"] = ["kimi-k2p7-code"]
    config["model_policy"]["factual"] = ["kimi-k2p7-code"]
    router = FrugalRouter(config=config, allow_remote=True)

    class PrefixOnlyProvider:
        def answer(self, task, model=None, category="general"):
            if not model.startswith("accounts/fireworks/models/"):
                raise FireworksError("not found", status_code=404)
            return Answer(text="ok", provider="fireworks", model=model)

    router.remote = PrefixOnlyProvider()

    result = router.run(Task(id="test", input="Explain what ROCm is."))

    assert result.route == "remote"
    assert result.answer.model == "accounts/fireworks/models/kimi-k2p7-code"


def test_router_retries_rate_limited_model_with_backoff():
    config = load_config("config/models.json")
    config["remote_backoff_seconds"] = 0.001
    router = FrugalRouter(config=config, allow_remote=True)

    calls = {"count": 0}

    class RateLimitedOnceProvider:
        def answer(self, task, model=None, category="general"):
            calls["count"] += 1
            if calls["count"] == 1:
                raise FireworksError("rate limited", status_code=429)
            return Answer(text="recovered", provider="fireworks", model=model or "fake")

    router.remote = RateLimitedOnceProvider()

    result = router.run(Task(id="test", input="Explain what ROCm is."))

    assert result.route == "remote"
    assert result.answer.text == "recovered"
    assert calls["count"] == 2


def test_router_moves_to_next_model_after_exhausting_retries():
    config = load_config("config/models.json")
    config["remote_backoff_seconds"] = 0.001
    config["remote_max_attempts"] = 2
    config["allowed_models"] = ["always-limited", "healthy-model"]
    config["model_policy"]["factual"] = ["always-limited", "healthy-model"]
    router = FrugalRouter(config=config, allow_remote=True)

    class SplitProvider:
        def answer(self, task, model=None, category="general"):
            if model == "always-limited":
                raise FireworksError("rate limited", status_code=429)
            return Answer(text="ok", provider="fireworks", model=model)

    router.remote = SplitProvider()

    result = router.run(Task(id="test", input="Explain what ROCm is."))

    assert result.route == "remote"
    assert result.answer.model == "healthy-model"


def test_router_rescues_truncated_nonempty_answer():
    config = load_config("config/models.json")
    config["remote_backoff_seconds"] = 0.001
    router = FrugalRouter(config=config, allow_remote=True)

    class SpilledReasoningProvider:
        def answer(self, task, model=None, category="general", max_tokens_override=None):
            if max_tokens_override is None:
                return Answer(
                    text="We need answer user. They want",
                    provider="fireworks",
                    model=model or "fake",
                    finish_reason="length",
                )
            return Answer(text="def fixed(): pass", provider="fireworks", model=model or "fake", finish_reason="stop")

    router.remote = SpilledReasoningProvider()

    result = router.run(Task(id="test", input="Debug this code:\n\ndef broken(): pass"))

    assert result.route == "remote"
    assert result.answer.text == "def fixed(): pass"


def test_router_rescues_empty_answer_at_token_cap():
    config = load_config("config/models.json")
    config["remote_backoff_seconds"] = 0.001
    router = FrugalRouter(config=config, allow_remote=True)

    seen_overrides = []

    class EmptyAtCapProvider:
        def answer(self, task, model=None, category="general", max_tokens_override=None):
            seen_overrides.append(max_tokens_override)
            if max_tokens_override is None:
                return Answer(text="", provider="fireworks", model=model or "fake", finish_reason="length")
            return Answer(text="rescued", provider="fireworks", model=model or "fake", finish_reason="stop")

    router.remote = EmptyAtCapProvider()

    result = router.run(Task(id="test", input="Explain what ROCm is."))

    assert result.route == "remote"
    assert result.answer.text == "rescued"
    assert seen_overrides == [None, 900]


def test_model_policy_uses_allowed_models_from_harness():
    config = load_config("config/models.json")
    config["allowed_models"] = ["accounts/fireworks/models/kimi-k2p7-code", "accounts/fireworks/models/minimax-m3"]
    policy = ModelPolicy(config)

    assert policy.choose("code_generation") == "accounts/fireworks/models/kimi-k2p7-code"
    assert policy.choose("factual") == "accounts/fireworks/models/minimax-m3"


def test_track1_json_io_contract(tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    input_path.write_text(
        json.dumps([{"task_id": "t1", "prompt": "Calculate 2 + 2. Return only the number."}]),
        encoding="utf-8",
    )

    tasks = read_tasks(input_path)
    write_results(output_path, [{"task_id": tasks[0].id, "answer": "4", "route": "local"}])

    assert tasks[0].id == "t1"
    assert tasks[0].input.startswith("Calculate")
    assert json.loads(output_path.read_text(encoding="utf-8")) == [{"task_id": "t1", "answer": "4"}]
