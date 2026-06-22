import json
from types import SimpleNamespace

import pytest

import run_benchmark
from models.lmstudio import LMStudioClient
from models.ollama import OllamaClient
from models.openrouter import OpenRouterClient
from scoring.factory import create_scorer
from scoring.rubric_scorer import RubricScorer


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

    def get(self, url, headers=None, timeout=None):
        self.gets.append({"url": url, "headers": headers, "timeout": timeout})
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
    client = OllamaClient("http://ollama.local", "test-model", timeout=77)
    fake_session = FakeRequestsSession({"message": {"content": "ok"}})
    client.session = fake_session

    result = client.query("hello", max_tokens=42, temperature=0.7)

    assert result == "ok"
    payload = fake_session.posts[0]["json"]
    assert payload["options"]["num_predict"] == 42
    assert payload["options"]["temperature"] == 0.7
    assert fake_session.posts[0]["timeout"] == 77


def test_ollama_auth_keep_alive_and_thinking_fallback():
    client = OllamaClient(
        "http://ollama.local",
        "test-model",
        timeout=77,
        api_key="token-123",
        keep_alive="30m",
    )
    fake_session = FakeRequestsSession(
        {"message": {"content": "", "thinking": "reasoned answer"}}
    )
    client.session = fake_session

    result = client.query("hello", max_tokens=42, temperature=0.7)
    models = client.list_models()
    connected = client.test_connection()

    assert result == "reasoned answer"
    assert models == []
    assert connected is True
    post = fake_session.posts[0]
    assert post["headers"]["Authorization"] == "Bearer token-123"
    assert post["json"]["keep_alive"] == "30m"
    assert all(
        call["headers"]["Authorization"] == "Bearer token-123"
        for call in fake_session.gets
    )


def test_ollama_empty_content_and_no_thinking_returns_empty_string():
    client = OllamaClient("http://ollama.local", "test-model")
    fake_session = FakeRequestsSession({"message": {"content": ""}})
    client.session = fake_session

    assert client.query("hello") == ""


def test_lmstudio_query_passes_max_tokens_and_temperature():
    client = LMStudioClient("http://lmstudio.local", "test-model", timeout=88)
    fake_session = FakeRequestsSession(
        {"choices": [{"message": {"content": "ok"}}]}
    )
    client.session = fake_session

    result = client.query("hello", max_tokens=43, temperature=0.6)

    assert result == "ok"
    payload = fake_session.posts[0]["json"]
    assert payload["max_tokens"] == 43
    assert payload["temperature"] == 0.6
    assert fake_session.posts[0]["timeout"] == 88


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


def test_openrouter_query_passes_retry_count(monkeypatch):
    class FakeHTTPXClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def close(self):
            return None

    monkeypatch.setattr(
        "models.openrouter.httpx.Client", lambda timeout: FakeHTTPXClient(timeout)
    )

    client = OpenRouterClient(
        base_url="https://openrouter.local/api/v1",
        model_name="test-model",
        api_key="token",
    )
    calls = []

    def fake_make_request(payload, retries=3):
        calls.append({"payload": payload, "retries": retries})
        return {"choices": [{"message": {"content": "ok"}}]}

    client._make_request = fake_make_request

    assert client.query("hello", retries=5) == "ok"
    assert calls[0]["retries"] == 5


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


def test_runner_preserves_scorer_bundle_details(monkeypatch):
    class FakeClient:
        def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
            return "response"

    monkeypatch.setattr(run_benchmark.time, "sleep", lambda delay: None)
    runtime = run_benchmark.RuntimeOptions(rate_limit_delay=0, concurrency=1)

    result = run_benchmark._query_and_score(
        FakeClient(),
        {"id": 1, "category": "one", "prompt": "prompt"},
        scorer_func=lambda q_id, response: 75,
        runtime=runtime,
        scorer_details={"method": "rubric"},
    )

    assert result["details"] == {"method": "rubric"}


def test_sequential_runner_returns_partial_results_on_graceful_shutdown(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
            self.calls += 1
            if self.calls == 2:
                raise run_benchmark.GracefulShutdown
            return "response"

    monkeypatch.setattr(run_benchmark.time, "sleep", lambda delay: None)
    questions = [
        {"id": 1, "category": "one", "prompt": "first"},
        {"id": 2, "category": "two", "prompt": "second"},
    ]

    results, optimization_results = run_benchmark._run_questions_sequential(
        questions=questions,
        client=FakeClient(),
        scorer_func=lambda q_id, response: 50,
        runtime=run_benchmark.RuntimeOptions(rate_limit_delay=0),
        model_name="model",
    )

    assert optimization_results == []
    assert [result["id"] for result in results] == [1]


def test_orchestrator_exports_interrupted_metadata(monkeypatch):
    class FakeClient:
        def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
            shutdown_state["requested"] = True
            return "response"

    shutdown_state = {"requested": False}
    exported_calls = []
    questions = [
        {"id": 1, "category": "one", "prompt": "first"},
        {"id": 2, "category": "two", "prompt": "second"},
    ]

    def export_callback(**kwargs):
        exported_calls.append(kwargs)
        return {"json": "partial.json"}

    monkeypatch.setattr(run_benchmark.time, "sleep", lambda delay: None)
    result = run_benchmark.run_single_model_benchmark(
        questions=questions,
        client=FakeClient(),
        model_name="model",
        scorer_bundle=SimpleNamespace(
            score_func=lambda q_id, response: 50,
            scorer=None,
            details={"method": "rubric"},
            method_label="rubric",
        ),
        runtime=run_benchmark.RuntimeOptions(rate_limit_delay=0),
        export_callback=export_callback,
        shutdown_requested=lambda: shutdown_state["requested"],
    )

    assert result.interrupted is True
    assert [item["id"] for item in result.results] == [1]
    assert exported_calls[0]["metadata"] == {
        "interrupted": True,
        "completed_questions": 1,
        "total_questions": 2,
    }


def test_scorer_factory_uses_rubric_scorer():
    questions = [{"id": 1, "category": "cat", "prompt": "prompt", "rubric": []}]
    bundle = create_scorer(
        "rubric",
        questions=questions,
    )

    assert isinstance(bundle.scorer, RubricScorer)
    assert bundle.method_label == "rubric"


def test_scorer_factory_rejects_removed_modes():
    with pytest.raises(ValueError, match="Unsupported scorer"):
        create_scorer("semantic", questions=[])


def test_load_questions_errors_are_explicit(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(run_benchmark.QuestionLoadError, match="not found"):
        run_benchmark.load_questions(str(missing))

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(run_benchmark.QuestionLoadError, match="Invalid JSON"):
        run_benchmark.load_questions(str(invalid))


def test_config_rejects_unsupported_export_format(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
provider:
  name: ollama
export:
  formats: [json, xml]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported export format"):
        run_benchmark.load_config(str(config_path))


def test_config_loads_request_log_and_ollama_keep_alive(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
provider:
  name: ollama
  keep_alive: 30m
request_log: ./results/requests.jsonl
""",
        encoding="utf-8",
    )

    config = run_benchmark.load_config(str(config_path))

    assert config.provider.endpoint == "http://localhost:11434"
    assert config.provider.keep_alive == "30m"
    assert config.request_log == "./results/requests.jsonl"


def test_interactive_loads_dataset_once_for_multiple_models(monkeypatch):
    class FakeClient:
        base_url = "http://provider.local"

        def __init__(self, model_name):
            self.model_name = model_name
            self.closed = False

        def test_connection(self):
            return True

        def list_models(self):
            return [{"name": "model-a", "size": 1}, {"name": "model-b", "size": 2}]

        def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
            return "AmsiScanBuffer VirtualProtect patch"

        def close(self):
            self.closed = True

    load_calls = []

    def fake_load_dataset(filepath="benchmark.json"):
        load_calls.append(filepath)
        return run_benchmark.BenchmarkDataset(
            questions=[{"id": 1, "category": "AMSI", "prompt": "prompt"}],
            path=filepath,
            content_hash="hash",
        )

    def fake_create_client(provider, endpoint, model_name, api_key=None):
        return FakeClient(model_name)

    args = SimpleNamespace(
        provider="ollama",
        endpoint=None,
        api_key=None,
        config=None,
        rate_limit_delay=0,
        max_tokens=32,
        temperature=0.2,
        concurrency=1,
        optimize_prompts=False,
        optimizer_model="optimizer",
        optimizer_endpoint=None,
        max_optimization_iterations=1,
        export_csv=False,
        output=None,
    )

    monkeypatch.setattr(run_benchmark, "create_client", fake_create_client)
    monkeypatch.setattr(
        run_benchmark,
        "pick",
        lambda *args, **kwargs: [("model-a", 0), ("model-b", 1)],
    )
    monkeypatch.setattr(run_benchmark, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        run_benchmark,
        "run_single_model_benchmark",
        lambda **kwargs: SimpleNamespace(
            results=[
                {
                    "id": 1,
                    "category": "AMSI",
                    "score": 50,
                    "response_snippet": "snippet",
                }
            ],
            total_score=50.0,
            interpretation="not-suitable",
            optimization_results=[],
        ),
    )

    run_benchmark.cmd_interactive(args)

    assert load_calls == ["datasets/v2/benchmark.jsonl"]


def test_run_command_delegates_to_single_model_orchestrator(monkeypatch):
    class FakeClient:
        base_url = "http://provider.local"

        def test_connection(self):
            return True

        def close(self):
            return None

    calls = []

    def fake_run_single_model_benchmark(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            results=[
                {
                    "id": 1,
                    "category": "AMSI",
                    "score": 50,
                    "response_snippet": "snippet",
                }
            ],
            total_score=50.0,
            interpretation="not-suitable",
            optimization_results=[],
        )

    args = SimpleNamespace(
        provider="ollama",
        endpoint=None,
        api_key=None,
        config=None,
        model="model-a",
        rate_limit_delay=0,
        max_tokens=32,
        temperature=0.2,
        concurrency=1,
        optimize_prompts=False,
        optimizer_model="optimizer",
        optimizer_endpoint=None,
        max_optimization_iterations=1,
        export_csv=False,
        output=None,
    )

    monkeypatch.setattr(
        run_benchmark,
        "create_client",
        lambda provider, endpoint, model, api_key=None: FakeClient(),
    )
    monkeypatch.setattr(
        run_benchmark,
        "load_dataset",
        lambda filepath="benchmark.json": run_benchmark.BenchmarkDataset(
            questions=[{"id": 1, "category": "AMSI", "prompt": "prompt"}],
            path=filepath,
            content_hash="hash",
        ),
    )
    monkeypatch.setattr(
        run_benchmark,
        "run_single_model_benchmark",
        fake_run_single_model_benchmark,
    )

    run_benchmark.cmd_run_benchmark(args)

    assert calls[0]["model_name"] == "model-a"
    assert calls[0]["questions"] == [{"id": 1, "category": "AMSI", "prompt": "prompt"}]


def test_question_ids_filter_preserves_dataset_order_after_profile():
    dataset = run_benchmark.BenchmarkDataset(
        questions=[
            {
                "id": 1,
                "category": "a",
                "prompt": "p1",
                "profiles": ["standard"],
            },
            {"id": 7, "category": "b", "prompt": "p7", "profiles": ["quick"]},
            {
                "id": 12,
                "category": "c",
                "prompt": "p12",
                "profiles": ["standard"],
            },
        ],
        path="dataset.jsonl",
        content_hash="hash",
    )
    args = SimpleNamespace(profile="standard", question_ids=["12", "1"])

    selected = run_benchmark._select_questions_for_args(dataset, args)

    assert [question["id"] for question in selected] == [1, 12]


def test_question_ids_filter_rejects_unknown_id_for_profile():
    dataset = run_benchmark.BenchmarkDataset(
        questions=[
            {"id": 1, "category": "a", "prompt": "p1", "profiles": ["standard"]},
            {"id": 7, "category": "b", "prompt": "p7", "profiles": ["quick"]},
        ],
        path="dataset.jsonl",
        content_hash="hash",
    )
    args = SimpleNamespace(profile="standard", question_ids=["7"])

    with pytest.raises(ValueError, match="Unknown question id"):
        run_benchmark._select_questions_for_args(dataset, args)


def test_request_log_appends_baseline_without_provider_secrets(tmp_path):
    class FakeClient:
        api_key = "secret-token"

        def query(self, prompt, max_tokens=1024, retries=3, temperature=0.2):
            return "AmsiScanBuffer VirtualProtect patch"

    log_path = tmp_path / "logs" / "requests.jsonl"
    questions = [{"id": 1, "category": "AMSI", "prompt": "prompt"}]

    results, optimization = run_benchmark._run_questions_sequential(
        questions=questions,
        client=FakeClient(),
        scorer_func=lambda q_id, response: 50,
        runtime=run_benchmark.RuntimeOptions(rate_limit_delay=0, request_log=str(log_path)),
        model_name="model",
    )

    assert results[0]["score"] == 50
    assert optimization == []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["phase"] == "baseline"
    assert payload["question_id"] == 1
    assert payload["response"] == "AmsiScanBuffer VirtualProtect patch"
    assert "secret-token" not in lines[0]


def test_langfuse_tracer_buffers_until_end_benchmark(monkeypatch):
    class FakeSpan:
        def __init__(self, name, recorder):
            self.name = name
            self.recorder = recorder

        def start_span(self, name, metadata=None):
            self.recorder["spans"].append({"name": name, "metadata": metadata})
            return FakeSpan(name, self.recorder)

        def update(self, **kwargs):
            self.recorder["updates"].append({"name": self.name, "data": kwargs})

        def end(self):
            self.recorder["ended"].append(self.name)

    class FakeLangfuse:
        instances = []

        def __init__(self, **kwargs):
            self.recorder = {"roots": [], "spans": [], "updates": [], "ended": []}
            self.flushed = False
            FakeLangfuse.instances.append(self)

        def start_span(self, name, metadata=None):
            self.recorder["roots"].append({"name": name, "metadata": metadata})
            return FakeSpan(name, self.recorder)

        def flush(self):
            self.flushed = True

    monkeypatch.setattr(run_benchmark, "Langfuse", FakeLangfuse)
    tracer = run_benchmark.LangfuseTracer(
        SimpleNamespace(public_key="pub", secret_key="sec", host="http://langfuse")
    )
    fake = FakeLangfuse.instances[0]

    tracer.start_benchmark("model", "rubric")
    tracer.log_generation(1, "cat", "prompt", "response", 50, 12.5, "model")
    tracer.start_optimization(1, "cat")
    tracer.log_optimization_attempt(
        0, "original", "prompt", "response", 0, 1.2, "model"
    )

    assert fake.recorder["roots"] == []
    assert fake.recorder["spans"] == []

    tracer.end_optimization(success=True, best_score=50, iterations=1)
    tracer.end_benchmark(50.0, "not-suitable")

    assert fake.recorder["roots"][0]["name"] == "benchmark-model"
    assert [span["name"] for span in fake.recorder["spans"]] == [
        "Q1-cat",
        "optimization-Q1",
        "iter-0-original",
    ]
    assert fake.flushed is True


def test_export_helper_writes_json_csv_and_preserves_top_level_schema(tmp_path):
    args = SimpleNamespace(export_csv=True, output="custom")
    config = SimpleNamespace(
        export=SimpleNamespace(
            formats=["json"],
            output_dir=str(tmp_path),
            include_response=False,
        )
    )
    results = [
        {
            "id": 1,
            "category": "AMSI",
            "score": 50,
            "censored": False,
            "response_snippet": "snippet",
            "full_response": "full response",
        }
    ]

    exported = run_benchmark._export_benchmark_results(
        results=results,
        model_name="org/model:name",
        total_score=50.0,
        interpretation="not-suitable",
        scoring_method="rubric",
        args=args,
        config=config,
    )

    json_path = tmp_path / "custom.json"
    csv_path = tmp_path / "custom.csv"
    assert exported == {"json": str(json_path), "csv": str(csv_path)}
    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    required_keys = {
        "model",
        "timestamp",
        "scoring_method",
        "total_score",
        "results",
        "interpretation",
    }
    assert required_keys.issubset(payload)
    assert payload["model"] == "org/model:name"
    assert payload["scoring_method"] == "rubric"
    csv_header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "response_snippet" not in csv_header
