# Red Team AI Benchmark

**Russian version:** [README.ru.md](README.ru.md)

Red Team AI Benchmark is a CLI benchmark for choosing base LLMs for authorized red-team and offensive-security work. Version 2 uses a rubric-based dataset instead of judging answers only against one golden response.

The default v2 suite contains 60 questions in `datasets/v2/benchmark.jsonl`, grouped by domain and difficulty.

## v2 Local Leaderboard

Top local models from the June 2026 v2 run, sorted by `judge_adjusted_score`.
The run used the full `standard` profile, Ollama, `max_tokens=4096`, `temperature=0.2`,
and post-hoc disputed-case LLM-as-Judge via OpenRouter `deepseek/deepseek-v4-flash`.

| Rank | Model | Rubric | Judge-adjusted | Judge critical error rate |
| --- | --- | ---: | ---: | ---: |
| 1 | `hf.co/BugTraceAI/BugTraceAI-Apex-G4-26B-Q4:latest` | `80.89%` | `89.45%` | `0.00%` |
| 2 | `nemotron-3-nano:30b` | `75.55%` | `86.81%` | `7.14%` |
| 3 | `hf.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q6_K` | `73.23%` | `81.12%` | `7.14%` |
| 4 | `hf.co/unsloth/Qwen3-Coder-Next-GGUF:Qwen3-Coder-Next-Q4_K_S` | `75.50%` | `80.15%` | `33.33%` |
| 5 | `mistral-small3.2:24b` | `69.39%` | `76.58%` | `8.33%` |

`judge_adjusted_score` is the recommended audit score for this leaderboard: the
deterministic rubric score with judged disputed cases replaced by the LLM-as-Judge
score. Keep the critical-error rate in view when choosing a base model.

## What v2 Measures

The benchmark reports the total weighted score and separate audit metrics:

| Metric | Meaning |
| --- | --- |
| `refusal_rate` | Percentage of refused or censored answers |
| `technical_accuracy` | Average rubric accuracy for technical criteria |
| `critical_error_rate` | Answers with fatal technical falsehoods |
| `completeness` | Coverage of required steps and conditions |
| `specificity` | Presence of concrete tools, fields, commands, or evidence |
| `hallucination_rate` | Currently tied to critical technical errors |
| `latency_ms_avg` | Average response latency |

Interpretation labels are deliberately conservative:

| Final score | Interpretation |
| --- | --- |
| `< 60%` | `not-suitable` |
| `60-79.9%` | `requires-validation` |
| `>= 80%` | `strong-candidate` |

A high score is not a production approval. Review domain, difficulty, refusal, and critical-error breakdowns before selecting a model.

## Dataset Coverage

The v2 dataset covers:

- Windows tradecraft
- AD and AD CS
- Web exploitation
- Cloud and IAM
- Containers and Kubernetes
- Detection and evasion reasoning
- OpSec and operational tradeoffs
- Tool usage
- Post-exploitation planning
- Validation and reporting

Difficulty levels are `L1 factual`, `L2 procedure`, `L3 troubleshooting`, `L4 scenario reasoning`, and `L5 multi-step operator task`.

## Installation

Requirements:

- Python `3.13+`
- `uv`
- One provider: Ollama, LM Studio, OpenWebUI, or OpenRouter

Install base dependencies:

```bash
uv sync
```

## Providers

| Provider | Default endpoint | Notes |
| --- | --- | --- |
| `ollama` | `http://localhost:11434` | Native Ollama API; optional Bearer auth for reverse proxies |
| `lmstudio` | `http://localhost:1234` | OpenAI-compatible LM Studio API |
| `openwebui` | `http://localhost:3000` | OpenAI-compatible OpenWebUI API |
| `openrouter` | `https://openrouter.ai/api/v1` | Requires an API key |

## Usage

List models:

```bash
uv run run_benchmark.py ls ollama
uv run run_benchmark.py ls lmstudio
uv run run_benchmark.py ls openwebui
uv run run_benchmark.py ls openrouter --api-key "$OPENROUTER_API_KEY"
```

Run the default v2 standard profile:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b"
```

Run a quick smoke subset:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --profile quick
```

Run selected v2 questions by ID:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --question-ids 5 12
```

Write an append-only per-question request log:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --request-log results/requests.jsonl
```

Run multiple local models interactively:

```bash
uv run run_benchmark.py interactive ollama --profile standard
```

Supported profiles:

| Profile | Purpose |
| --- | --- |
| `quick` | 16-question smoke subset |
| `standard` | Full 60-question v2 benchmark |
| `enterprise` | Full v2 dataset with audit-friendly export |
| `local-only` | Full v2 dataset without LLM judge usage |
| `cloud-comparison` | Full v2 dataset for fixed cloud-model comparisons |

## Scoring

Runtime scoring is always `rubric`. It is deterministic and does not require an external LLM judge. Each v2 question contains atomic criteria, fatal-error patterns, acceptable variants, tags, and a question weight.

Runtime scoring does not support legacy `keyword`, `semantic`, or `hybrid` modes. Use the offline `judge` command for post-hoc LLM-as-Judge auditing.

## Offline LLM-as-Judge

Saved v2 result JSON files can be audited post-hoc without rerunning benchmark models:

```bash
OPENROUTER_API_KEY=... uv run run_benchmark.py judge \
  --results "results_*_v2/*.json" \
  --dataset datasets/v2/benchmark.jsonl \
  --judge-model "deepseek/deepseek-v4-flash" \
  --output-dir judge_results_v2 \
  --mode disputed \
  --concurrency 4
```

The judge command writes `per_model/*.json`, `detailed.csv`, `summary.csv`, and `disputed_cases.csv`. `judge_score` is the judged subset score; use `judge_adjusted_score` for the rubric score with judged disputed cases replaced. LLM-as-Judge output is an audit layer and does not overwrite deterministic benchmark results.

## Configuration

Copy `config.example.yaml` to `config.yaml` and adjust it:

```yaml
provider:
  name: ollama
  endpoint: http://localhost:11434
  # api_key: sk-xxx
  # keep_alive: 30m

scoring:
  method: rubric

export:
  formats:
    - json
    - csv
    - criteria_csv
  output_dir: ./results
  include_response: true

questions_file: datasets/v2/benchmark.jsonl
answers_file: answers_all.txt
rate_limit_delay: 1.5
max_tokens: 1024
temperature: 0.2
concurrency: 1
# request_log: ./results/requests.jsonl
```

Run with config:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --config config.yaml
```

## Output

JSON export includes model results, per-question rubric evidence, aggregate summary, and audit provenance:

```json
{
  "model": "llama3.1:8b",
  "scoring_method": "rubric",
  "total_score": 75.0,
  "interpretation": "requires-validation",
  "benchmark_version": "2.1.0",
  "dataset_id": "redteam-ai-benchmark-v2",
  "dataset_version": "2.0.0",
  "dataset_hash": "...",
  "scorer_version": "rubric",
  "config_hash": "...",
  "git_commit": "...",
  "package_version": "2.1.0",
  "runtime_profile": "standard",
  "summary": {
    "metrics": {
      "refusal_rate": 0.0,
      "critical_error_rate": 0.0
    },
    "breakdown": {
      "difficulty": {},
      "domain": {},
      "capability": {}
    }
  }
}
```

CSV output contains per-question rows. `criteria_csv` adds one row per passed or failed rubric criterion.

## Prompt Optimization

Prompt optimization remains optional and separate from base-model scoring. It only runs for `0%` censored responses when `--optimize-prompts` is enabled, and it writes `optimized_prompts_{model}_{timestamp}.json`.

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" \
  --optimize-prompts \
  --optimizer-model "llama3.3:70b"
```

Do not mix optimized scores with base model capability comparisons.

## Validation

Useful checks:

```bash
uv run run_benchmark.py --help
uv run run_benchmark.py run --help
uv run pytest
python3 -m compileall -q run_benchmark.py benchmark models scoring utils
```

## License

MIT. Use in authorized red team labs, commercial security assessments, AI-security research, and educational environments.
