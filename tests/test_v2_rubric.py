import json
from pathlib import Path

import pytest

import run_benchmark
from benchmark.io import QuestionLoadError, load_dataset
from benchmark.metrics import summarize_results, weighted_score
from scoring.factory import create_scorer
from scoring.rubric_scorer import RubricScorer


def test_v2_dataset_loads_and_validates_contract():
    dataset = load_dataset("datasets/v2/benchmark.jsonl")

    assert dataset.metadata["schema"] == "rubric-v2"
    assert dataset.metadata["dataset_version"] == "2.0.0"
    assert len(dataset.questions) == 60
    assert dataset.content_hash
    assert all("category" in question for question in dataset.questions)
    assert {question["domain"] for question in dataset.questions} >= {
        "windows-tradecraft",
        "ad-adcs",
        "web-exploitation",
        "cloud-iam",
        "container-kubernetes",
        "detection-evasion",
        "opsec-tradeoff",
        "tool-usage",
        "post-exploitation",
        "validation-reporting",
    }


def test_v2_dataset_validation_errors_are_explicit(tmp_path):
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        "\n".join([
            json.dumps({"record_type": "manifest", "schema": "rubric-v2"}),
            json.dumps({"record_type": "question", "id": 1, "prompt": "missing"}),
        ]),
        encoding="utf-8",
    )

    with pytest.raises(QuestionLoadError, match="missing required fields"):
        load_dataset(str(invalid))

    duplicate = tmp_path / "duplicate.jsonl"
    dataset = load_dataset("datasets/v2/benchmark.jsonl")
    first = dataset.questions[0]
    duplicate.write_text(
        "\n".join([
            json.dumps({"record_type": "manifest", "schema": "rubric-v2"}),
            json.dumps(first),
            json.dumps(first),
        ]),
        encoding="utf-8",
    )

    with pytest.raises(QuestionLoadError, match="Duplicate question id"):
        load_dataset(str(duplicate))


def test_rubric_scorer_calibration_pack_orders_quality_levels():
    dataset = load_dataset("datasets/v2/benchmark.jsonl")
    responses = json.loads(
        Path("tests/fixtures/calibration/rubric_responses.json").read_text(
            encoding="utf-8",
        )
    )
    scorer = RubricScorer(dataset.questions)
    q_id = responses["question_id"]

    correct = scorer.score(q_id, responses["correct"])
    partial = scorer.score(q_id, responses["partial"])
    vague = scorer.score(q_id, responses["vague"])
    refusal = scorer.score(q_id, responses["refusal"])
    fatal = scorer.score(q_id, responses["fatal"])

    assert correct.score >= 80
    assert correct.criteria_passed
    assert partial.score < correct.score
    assert vague.score < partial.score
    assert refusal.score == 0
    assert refusal.censored is True
    assert fatal.score == 0
    assert fatal.critical_error is True


def test_rubric_scorer_factory_is_default_capable():
    dataset = load_dataset("datasets/v2/benchmark.jsonl")
    bundle = create_scorer(
        "rubric",
        questions=dataset.questions,
    )

    assert isinstance(bundle.scorer, RubricScorer)
    result = bundle.scorer.score(
        55,
        "The report needs rubric criteria, passed and failed evidence, model config, dataset version, scorer version, and response snippet.",
    )
    assert result.score >= 80
    assert result.details["method"] == "rubric"


def test_weighted_summary_breaks_down_v2_results():
    results = [
        {
            "score": 100,
            "weight": 2,
            "domain": "a",
            "difficulty": "L1 factual",
            "capability": "x",
            "censored": False,
            "critical_error": False,
            "metrics": {"technical_accuracy": 1.0},
            "latency_ms": 10,
        },
        {
            "score": 0,
            "weight": 1,
            "domain": "b",
            "difficulty": "L5 multi-step operator task",
            "capability": "y",
            "censored": True,
            "critical_error": True,
            "metrics": {"technical_accuracy": 0.0},
            "latency_ms": 20,
        },
    ]

    summary = summarize_results(results)
    assert weighted_score(results) == pytest.approx(66.666, rel=0.01)
    assert summary["weighted_score"] == 66.67
    assert summary["metrics"]["refusal_rate"] == 50.0
    assert summary["metrics"]["critical_error_rate"] == 50.0
    assert summary["breakdown"]["domain"]["a"]["score"] == 100.0


def test_export_includes_v2_provenance_and_criteria_csv(tmp_path):
    dataset = load_dataset("datasets/v2/benchmark.jsonl")
    args = type(
        "Args",
        (),
        {
            "export_csv": False,
            "output": "v2-result",
            "profile": "standard",
            "provider": "ollama",
            "endpoint": "http://localhost:11434",
            "config": None,
            "optimize_prompts": False,
            "optimizer_model": None,
            "optimizer_endpoint": None,
            "max_optimization_iterations": None,
            "api_key": "secret-value",
        },
    )()
    config = type(
        "Config",
        (),
        {
            "export": type(
                "Export",
                (),
                {
                    "formats": ["json", "criteria_csv"],
                    "output_dir": str(tmp_path),
                    "include_response": True,
                },
            )()
        },
    )()
    runtime = run_benchmark.RuntimeOptions(
        rate_limit_delay=0,
        max_tokens=32,
        temperature=0.3,
        concurrency=2,
    )
    results = [
        {
            "id": 55,
            "category": "Finding evidence",
            "score": 100,
            "response_snippet": "snippet",
            "full_response": "full",
            "censored": False,
            "criteria_passed": ["criteria"],
            "criteria_failed": [],
            "domain": "validation-reporting",
            "capability": "Finding evidence",
            "difficulty": "L1 factual",
            "weight": 1.0,
        }
    ]

    exported = run_benchmark._export_benchmark_results(
        results=results,
        model_name="model",
        total_score=100.0,
        interpretation="strong-candidate",
        scoring_method="rubric",
        args=args,
        config=config,
        dataset=dataset,
        runtime=runtime,
        summary={"metrics": {"refusal_rate": 0.0}, "breakdown": {}},
    )

    payload = json.loads((tmp_path / "v2-result.json").read_text(encoding="utf-8"))
    assert exported["json"].endswith("v2-result.json")
    assert exported["criteria_csv"].endswith("v2-result_criteria.csv")
    assert payload["benchmark_version"] == "2.1.0"
    assert payload["dataset_id"] == "redteam-ai-benchmark-v2"
    assert payload["dataset_hash"] == dataset.content_hash
    assert payload["runtime_profile"] == "standard"
    assert payload["run_config"]["provider"] == "ollama"
    assert payload["run_config"]["model"] == "model"
    assert payload["run_config"]["profile"] == "standard"
    assert payload["run_config"]["questions_file"] == dataset.path
    assert payload["run_config"]["max_tokens"] == 32
    assert payload["run_config"]["temperature"] == 0.3
    assert payload["run_config"]["rate_limit_delay"] == 0
    assert payload["run_config"]["concurrency"] == 2
    assert "api_key" not in payload["run_config"]
    assert payload["summary"]["metrics"]["refusal_rate"] == 0.0
    assert (tmp_path / "v2-result_criteria.csv").exists()
