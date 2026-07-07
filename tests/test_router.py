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
            if model == "missing-model":
                raise FireworksError("not found", status_code=404)
            return Answer(text=f"{model}:ok", provider="fireworks", model=model or "fake")

    router.remote = FlakyProvider()

    result = router.run(Task(id="test", input="Explain what ROCm is."))

    assert result.route == "remote"
    assert result.used_remote is True
    assert result.answer.model == "accounts/fireworks/models/minimax-m3"


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
