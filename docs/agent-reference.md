# Agent Reference

This document keeps implementation details out of `AGENTS.md` and user-facing README files.

## v2 Dataset

Default dataset:

```text
datasets/v2/benchmark.jsonl
```

The first JSONL record is a manifest with `schema: rubric-v2`, `dataset_id`, `dataset_version`, and `benchmark_version`. Every question record must include:

- `id`
- `domain`
- `capability`
- `difficulty`
- `prompt`
- `expected_artifacts`
- `rubric`
- `fatal_errors`
- `acceptable_variants`
- `tags`
- `weight`

Allowed difficulty values:

- `L1 factual`
- `L2 procedure`
- `L3 troubleshooting`
- `L4 scenario reasoning`
- `L5 multi-step operator task`

Question loading is implemented in `benchmark/io.py`.

## Runtime Profiles

CLI profiles are defined in `run_benchmark.py`:

- `quick`: quick v2 smoke subset selected by question `profiles`.
- `standard`: default full v2 run.
- `enterprise`: full v2 run with audit-friendly export.
- `local-only`: full v2 run without LLM judge usage.
- `cloud-comparison`: full v2 run for fixed hosted-model comparisons.

If adding a profile, update `PROFILE_DEFAULTS`, CLI docs, README files, and tests.

`--question-ids` is an additional runtime filter. Apply it after profile filtering and preserve dataset order. Unknown IDs in the selected profile must fail before any model request.

## Scoring

Default scorer:

```text
rubric
```

`scoring/rubric_scorer.py` is deterministic and local. It checks each criterion pattern against the response, records passed and failed criteria, collects evidence, and applies fatal-error rules before normal scoring.

Do not add runtime scorer modes for keyword, semantic, hybrid, or online LLM judging. LLM-as-Judge belongs only in the offline `judge` command.

The request log is an optional JSONL side artifact selected with `--request-log` or top-level `request_log` in config. It may include prompts, responses, scores, latency, refusal and critical-error flags, and question metadata. It must not include provider headers or API keys.

## Offline LLM-as-Judge

Post-hoc judging is exposed through the main CLI:

```bash
uv run run_benchmark.py judge --results "results_*_v2/*.json" --mode disputed
```

The implementation lives in `benchmark/offline_judge.py`; do not add a separate script entrypoint. It reads saved v2 result JSON files, does not rerun benchmark models, and writes sidecar audit artifacts under the configured output directory. Treat `judge_score` as the judged subset score and `judge_adjusted_score` as the comparison-friendly adjusted result.

`ScoringResult` now carries:

- `score`
- `normalized_score`
- `censored`
- `critical_error`
- `criteria_passed`
- `criteria_failed`
- `evidence`
- `metrics`
- `details`

## Aggregation

`benchmark/metrics.py` owns aggregate scoring:

- `weighted_score(results)`
- `summarize_results(results)`

The benchmark exports:

- weighted total score
- metrics such as refusal and critical-error rate
- breakdowns by difficulty, domain, and capability

High scores are labeled `strong-candidate`, not `production-ready`.

## Export

`utils/export.py` writes JSON, CSV, and `criteria_csv`.

JSON exports include top-level audit provenance:

- `benchmark_version`
- `dataset_id`
- `dataset_version`
- `dataset_hash`
- `scorer_version`
- `config_hash`
- `git_commit`
- `package_version`
- `runtime_profile`

`criteria_csv` writes one row per passed or failed rubric criterion.

## Adding v2 Questions

When adding a question:

1. Add one JSON object to `datasets/v2/benchmark.jsonl`.
2. Keep `id` unique.
3. Use one allowed difficulty string.
4. Keep `rubric` non-empty and criteria weights positive.
5. Include fatal-error rules for common dangerous false claims.
6. Add or update calibration fixtures when scorer behavior changes.
7. Run `uv run pytest` and `python3 -m compileall -q run_benchmark.py benchmark models scoring utils`.

Do not add large batches of questions without rubric criteria.

## Optional Features

Prompt optimization remains separate from base-model scoring. Do not mix optimized results into base model comparison tables.

Langfuse tracing is optional and should not be required for local or CI validation.

Ollama supports optional reverse-proxy Bearer auth through `--api-key`, config, or `OLLAMA_API_KEY`, and optional `keep_alive` through `--ollama-keep-alive`, `provider.keep_alive`, or `OLLAMA_KEEP_ALIVE`. Keep those options provider-local.
