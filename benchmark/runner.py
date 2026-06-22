"""Benchmark question runners."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from scoring.base import ScoringResult
from utils.request_log import append_request_log

from .shutdown import GracefulShutdown
from .types import QueryResult, QuestionResult, RuntimeOptions


def _sleep_between_requests(rate_limit_delay: float) -> None:
    """Sleep only when rate limiting is explicitly positive."""
    if rate_limit_delay > 0:
        time.sleep(rate_limit_delay)


def _score_response(scorer_func, scorer, q_id: int, response: str) -> ScoringResult:
    """Score a response while preserving metadata when the scorer exposes it."""
    if scorer is not None and hasattr(scorer, "score"):
        result = scorer.score(q_id, response)
        if isinstance(result, ScoringResult):
            return result

    return ScoringResult(score=scorer_func(q_id, response))


def _with_default_details(
    scoring: ScoringResult, default_details: Optional[Dict[str, Any]]
) -> ScoringResult:
    """Attach scorer bundle details when the scorer does not return its own."""
    if default_details and not scoring.details:
        scoring.details = dict(default_details)
    return scoring


def _response_snippet(response: str, limit: int = 180) -> str:
    snippet = response[:limit].replace("\n", " ").strip()
    return snippet + ("..." if len(response) > limit else "")


def _make_result(
    q: Dict[str, Any],
    score: int,
    response: str,
    censored: bool = False,
    similarity: Optional[float] = None,
    latency_ms: Optional[float] = None,
    normalized_score: Optional[float] = None,
    critical_error: bool = False,
    criteria_passed: Optional[List[str]] = None,
    criteria_failed: Optional[List[str]] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the persisted per-question result object."""
    return QuestionResult(
        id=q["id"],
        category=q["category"],
        score=score,
        response_snippet=_response_snippet(response),
        full_response=response,
        censored=censored,
        similarity=similarity,
        latency_ms=latency_ms,
        normalized_score=normalized_score,
        critical_error=critical_error,
        criteria_passed=criteria_passed or [],
        criteria_failed=criteria_failed or [],
        evidence=evidence or [],
        metrics=metrics or {},
        difficulty=q.get("difficulty"),
        domain=q.get("domain"),
        capability=q.get("capability"),
        weight=float(q.get("weight", 1.0)),
        details=details or {},
    ).to_dict()


def _query_and_score(
    client,
    q: Dict[str, Any],
    scorer_func,
    runtime: RuntimeOptions,
    scorer=None,
    scorer_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Query a model for one question and score the response."""
    start_time = time.time()
    response = client.query(
        q["prompt"],
        max_tokens=runtime.max_tokens,
        temperature=runtime.temperature,
    )
    latency_ms = (time.time() - start_time) * 1000
    scoring = _with_default_details(
        _score_response(scorer_func, scorer, q["id"], response), scorer_details
    )
    return QueryResult(
        question=q,
        response=response,
        score=scoring.score,
        latency_ms=latency_ms,
        censored=scoring.censored,
        similarity=scoring.similarity,
        normalized_score=scoring.normalized_score,
        critical_error=scoring.critical_error,
        criteria_passed=scoring.criteria_passed,
        criteria_failed=scoring.criteria_failed,
        evidence=scoring.evidence,
        metrics=scoring.metrics,
        details=scoring.details,
    ).__dict__


def _log_request_result(
    runtime: RuntimeOptions,
    q: Dict[str, Any],
    *,
    phase: str,
    response: str,
    score: int,
    latency_ms: Optional[float],
    censored: bool,
    critical_error: bool,
    prompt: Optional[str] = None,
    optimization_iteration: Optional[int] = None,
    optimization_strategy: Optional[str] = None,
) -> None:
    """Write one request event without provider headers or credentials."""
    append_request_log(
        runtime.request_log,
        {
            "phase": phase,
            "question_id": q.get("id"),
            "category": q.get("category"),
            "domain": q.get("domain"),
            "difficulty": q.get("difficulty"),
            "capability": q.get("capability"),
            "prompt": prompt if prompt is not None else q.get("prompt"),
            "response": response,
            "score": score,
            "latency_ms": latency_ms,
            "censored": censored,
            "critical_error": critical_error,
            "optimization_iteration": optimization_iteration,
            "optimization_strategy": optimization_strategy,
        },
    )


def _run_questions_sequential(
    questions: List[Dict[str, Any]],
    client,
    scorer_func,
    runtime: RuntimeOptions,
    model_name: str,
    optimizer=None,
    reference_answers: Optional[Dict[int, str]] = None,
    tracer=None,
    scorer=None,
    scorer_details: Optional[Dict[str, Any]] = None,
    shutdown_requested: Optional[Callable[[], bool]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run questions sequentially, including optional optimization/tracing."""
    results = []
    optimization_results = []
    reference_answers = reference_answers or {}
    shutdown_requested = shutdown_requested or (lambda: False)

    for q in questions:
        if shutdown_requested():
            print("  ⚠️  Graceful shutdown before next question.")
            break

        print(f"[Q{q['id']:>2}] {q['category']}...")

        try:
            query_result = _query_and_score(
                client,
                q,
                scorer_func,
                runtime,
                scorer=scorer,
                scorer_details=scorer_details,
            )
        except GracefulShutdown:
            print("  ⚠️  Graceful shutdown during model query.")
            break
        response = query_result["response"]
        score = query_result["score"]
        latency_ms = query_result["latency_ms"]
        censored = query_result["censored"]
        similarity = query_result["similarity"]
        normalized_score = query_result["normalized_score"]
        critical_error = query_result["critical_error"]
        criteria_passed = query_result["criteria_passed"]
        criteria_failed = query_result["criteria_failed"]
        evidence = query_result["evidence"]
        metrics = query_result["metrics"]
        details = query_result["details"]

        _log_request_result(
            runtime,
            q,
            phase="baseline",
            response=response,
            score=score,
            latency_ms=latency_ms,
            censored=censored,
            critical_error=critical_error,
        )

        if score == 0 and optimizer:
            print("  ⚠️  Censored response (0%), starting optimization...")

            if tracer:
                tracer.start_optimization(q["id"], q["category"])

            try:
                opt_result = optimizer.optimize_prompt(
                    original_prompt=q["prompt"],
                    target_client=client,
                    scorer_func=scorer_func,
                    question_id=q["id"],
                    category=q["category"],
                    reference_answer=reference_answers.get(q["id"]),
                    initial_response=response,
                    initial_score=score,
                    max_tokens=runtime.max_tokens,
                    temperature=runtime.temperature,
                )
            except GracefulShutdown:
                print("  ⚠️  Graceful shutdown during prompt optimization.")
                break

            if tracer and opt_result.get("history"):
                for attempt in opt_result["history"]:
                    tracer.log_optimization_attempt(
                        iteration=attempt.get("iteration", 0),
                        strategy=attempt.get("strategy", "unknown"),
                        prompt=attempt.get("prompt", ""),
                        response=attempt.get("response", ""),
                        score=attempt.get("score", 0),
                        latency_ms=0,
                        model=model_name,
                    )

            for attempt in opt_result.get("history", []):
                if attempt.get("iteration") == 0:
                    continue
                _log_request_result(
                    runtime,
                    q,
                    phase="optimization",
                    prompt=attempt.get("prompt", q["prompt"]),
                    response=attempt.get("response", ""),
                    score=attempt.get("score", 0),
                    latency_ms=attempt.get("latency_ms"),
                    censored=attempt.get("score", 0) == 0,
                    critical_error=False,
                    optimization_iteration=attempt.get("iteration"),
                    optimization_strategy=attempt.get("strategy"),
                )

            if tracer:
                tracer.end_optimization(
                    success=opt_result["success"],
                    best_score=opt_result["score"],
                    iterations=opt_result["iterations"],
                )

            score = opt_result["score"]
            response = opt_result["response"]
            opt_scoring = _with_default_details(
                _score_response(scorer_func, scorer, q["id"], response),
                scorer_details,
            )
            censored = opt_scoring.censored
            similarity = opt_scoring.similarity
            normalized_score = opt_scoring.normalized_score
            critical_error = opt_scoring.critical_error
            criteria_passed = opt_scoring.criteria_passed
            criteria_failed = opt_scoring.criteria_failed
            evidence = opt_scoring.evidence
            metrics = opt_scoring.metrics
            details = opt_scoring.details

            optimization_results.append(
                {
                    "id": q["id"],
                    "category": q["category"],
                    "original_score": 0,
                    "best_score": score,
                    "best_prompt": opt_result["prompt"],
                    "iterations": opt_result["iterations"],
                    "success": opt_result["success"],
                    "optimization_attempts": opt_result["history"],
                }
            )

            print(f"  ✓ Optimization complete: {score}%\n")
        elif tracer:
            tracer.log_generation(
                question_id=q["id"],
                category=q["category"],
                prompt=q["prompt"],
                response=response,
                score=score,
                latency_ms=latency_ms,
                model=model_name,
            )

        results.append(
            _make_result(
                q,
                score,
                response,
                censored,
                similarity,
                latency_ms,
                normalized_score,
                critical_error,
                criteria_passed,
                criteria_failed,
                evidence,
                metrics,
                details,
            )
        )
        try:
            _sleep_between_requests(runtime.rate_limit_delay)
        except GracefulShutdown:
            print("  ⚠️  Graceful shutdown during rate-limit delay.")
            break

    return results, optimization_results


def _run_questions_concurrent(
    questions: List[Dict[str, Any]],
    client,
    scorer_func,
    runtime: RuntimeOptions,
    scorer=None,
    scorer_details: Optional[Dict[str, Any]] = None,
    shutdown_requested: Optional[Callable[[], bool]] = None,
) -> List[Dict[str, Any]]:
    """Run independent questions concurrently and return stable ordered results."""
    indexed_results = {}
    shutdown_requested = shutdown_requested or (lambda: False)
    interrupted = False

    executor = ThreadPoolExecutor(max_workers=runtime.concurrency)
    future_to_index = {}

    try:
        for index, q in enumerate(questions):
            if shutdown_requested():
                interrupted = True
                break
            if index > 0:
                try:
                    _sleep_between_requests(runtime.rate_limit_delay)
                except GracefulShutdown:
                    interrupted = True
                    break
            print(f"[Q{q['id']:>2}] {q['category']}...")
            future = executor.submit(
                _query_and_score,
                client,
                q,
                scorer_func,
                runtime,
                scorer,
                scorer_details,
            )
            future_to_index[future] = index

        for future in as_completed(future_to_index):
            if shutdown_requested():
                interrupted = True
                break
            index = future_to_index[future]
            try:
                query_result = future.result()
            except GracefulShutdown:
                interrupted = True
                break
            q = query_result["question"]
            _log_request_result(
                runtime,
                q,
                phase="baseline",
                response=query_result["response"],
                score=query_result["score"],
                latency_ms=query_result["latency_ms"],
                censored=query_result["censored"],
                critical_error=query_result["critical_error"],
            )
            indexed_results[index] = _make_result(
                q,
                query_result["score"],
                query_result["response"],
                query_result["censored"],
                query_result["similarity"],
                query_result["latency_ms"],
                query_result["normalized_score"],
                query_result["critical_error"],
                query_result["criteria_passed"],
                query_result["criteria_failed"],
                query_result["evidence"],
                query_result["metrics"],
                query_result["details"],
            )
    except GracefulShutdown:
        interrupted = True
    finally:
        if interrupted:
            print("  ⚠️  Graceful shutdown: cancelling pending benchmark questions.")
            for future in future_to_index:
                future.cancel()
        executor.shutdown(wait=True, cancel_futures=interrupted)

    return [indexed_results[index] for index in sorted(indexed_results)]


def _effective_concurrency(
    requested_concurrency: int,
    optimizer_enabled: bool,
    tracer_enabled: bool,
) -> int:
    """Disable concurrency for paths with shared mutable optimization/tracing state."""
    if requested_concurrency > 1 and (optimizer_enabled or tracer_enabled):
        print(
            "⚠️  Concurrency disabled because prompt optimization or Langfuse tracing is enabled"
        )
        return 1
    return requested_concurrency
