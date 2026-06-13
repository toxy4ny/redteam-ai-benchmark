from types import SimpleNamespace

import pytest

import run_benchmark
from models.lmstudio import LMStudioClient
from models.ollama import OllamaClient
from models.openrouter import OpenRouterClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeRequestsSession:
    def __init__(self, response_payload):
        self.response_payload = response_payload
        self.posts = []
        self.gets = []
        self.closed = False

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return FakeResponse(self.response_payload)

    def get(self, url, timeout=None):
        self.gets.append({"url": url, "timeout": timeout})
        return FakeResponse({"models": [], "data": []})

    def close(self):
        self.closed = True


def test_sleep_between_requests_skips_zero_delay(monkeypatch):
    calls = []
    monkeypatch.setattr(run_benchmark.time, "sleep", calls.append)

    run_benchmark._sleep_between_requests(0)
    run_benchmark._sleep_between_requests(0.25)

    assert calls == [0.25]


def test_runtime_options_cli_overrides_config():
    args = SimpleNamespace(
        rate_limit_delay=0,
        max_tokens=128,
        temperature=0.4,
        concurrency=3,
    )
    config = SimpleNamespace(
        rate_limit_delay=1.5,
        max_tokens=768,
        temperature=0.2,
        concurrency=1,
    )

    options = run_benchmark._resolve_runtime_options(args, config)

    assert options.rate_limit_delay == 0
    assert options.max_tokens == 128
    assert options.temperature == 0.4
    assert options.concurrency == 3


def test_ollama_query_passes_max_tokens_and_temperature():
    client = OllamaClient("http://ollama.local", "test-model")
    fake_session = FakeRequestsSession({"message": {"content": "ok"}})
    client.session = fake_session

    result = client.query("hello", max_tokens=42, temperature=0.7)

    assert result == "ok"
    payload = fake_session.posts[0]["json"]
    assert payload["options"]["num_predict"] == 42
    assert payload["options"]["temperature"] == 0.7


def test_lmstudio_query_passes_max_tokens_and_temperature():
    client = LMStudioClient("http://lmstudio.local", "test-model")
    fake_session = FakeRequestsSession(
        {"choices": [{"message": {"content": "ok"}}]}
    )
    client.session = fake_session

    result = client.query("hello", max_tokens=43, temperature=0.6)

    assert result == "ok"
    payload = fake_session.posts[0]["json"]
    assert payload["max_tokens"] == 43
    assert payload["temperature"] == 0.6


def test_openrouter_query_passes_max_tokens_and_temperature(monkeypatch):
    class FakeHTTPXClient:
        def __init__(self, timeout):
            self.timeout = timeout
            self.posts = []
            self.closed = False

        def post(self, url, headers=None, json=None):
            self.posts.append({"url": url, "headers": headers, "json": json})
            return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

        def get(self, url, headers=None):
            return FakeResponse({"data": [{"id": "test-model"}]})

        def close(self):
            self.closed = True

    fake_client = FakeHTTPXClient(timeout=120)
    monkeypatch.setattr(
        "models.openrouter.httpx.Client", lambda timeout: fake_client
    )

    client = OpenRouterClient(
        base_url="https://openrouter.local/api/v1",
        model_name="test-model",
        api_key="token",
    )
    result = client.query("hello", max_tokens=44, temperature=0.5)

    assert result == "ok"
    payload = fake_client.posts[0]["json"]
    assert payload["max_tokens"] == 44
    assert payload["temperature"] == 0.5

    client.close()
    assert fake_client.closed


def test_optimizer_reuses_initial_result_without_requerying_original_prompt():
    class FakeTargetClient:
        def __init__(self):
            self.calls = 0

        def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
            self.calls += 1
            return "unexpected"

    optimizer = run_benchmark.PromptOptimizer.__new__(run_benchmark.PromptOptimizer)
    optimizer.history = []
    optimizer.max_iterations = 0
    optimizer.min_acceptable_score = 50

    target_client = FakeTargetClient()
    result = optimizer.optimize_prompt(
        original_prompt="original",
        target_client=target_client,
        scorer_func=lambda q_id, response: pytest.fail("scorer should not run"),
        question_id=1,
        initial_response="already scored",
        initial_score=100,
        max_tokens=55,
        temperature=0.3,
    )

    assert target_client.calls == 0
    assert result["score"] == 100
    assert result["history"][0]["response"] == "already scored"


def test_optimizer_with_initial_censored_result_queries_only_optimized_prompt():
    class FakeTargetClient:
        def __init__(self):
            self.prompts = []

        def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
            self.prompts.append(
                {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            return "optimized response"

    optimizer = run_benchmark.PromptOptimizer.__new__(run_benchmark.PromptOptimizer)
    optimizer.history = []
    optimizer.max_iterations = 1
    optimizer.min_acceptable_score = 50
    optimizer._generate_optimized_variants = lambda **kwargs: {
        "role_playing": "optimized prompt"
    }

    target_client = FakeTargetClient()
    result = optimizer.optimize_prompt(
        original_prompt="original prompt",
        target_client=target_client,
        scorer_func=lambda q_id, response: 50,
        question_id=1,
        initial_response="refusal",
        initial_score=0,
        max_tokens=55,
        temperature=0.3,
    )

    assert target_client.prompts == [
        {"prompt": "optimized prompt", "max_tokens": 55, "temperature": 0.3}
    ]
    assert result["score"] == 50
    assert [attempt["prompt"] for attempt in result["history"]] == [
        "original prompt",
        "optimized prompt",
    ]


def test_concurrent_runner_preserves_question_order(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.calls = []

        def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
            self.calls.append(
                {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            if prompt == "slow":
                run_benchmark.time.sleep(0.02)
            return f"response-{prompt}"

    monkeypatch.setattr(run_benchmark.time, "sleep", lambda delay: None)
    questions = [
        {"id": 1, "category": "one", "prompt": "slow"},
        {"id": 2, "category": "two", "prompt": "fast"},
        {"id": 3, "category": "three", "prompt": "medium"},
    ]
    runtime = run_benchmark.RuntimeOptions(
        rate_limit_delay=0,
        max_tokens=99,
        temperature=0.8,
        concurrency=3,
    )

    results = run_benchmark._run_questions_concurrent(
        questions,
        FakeClient(),
        scorer_func=lambda q_id, response: q_id * 10,
        runtime=runtime,
    )

    assert [result["id"] for result in results] == [1, 2, 3]
    assert [result["score"] for result in results] == [10, 20, 30]
