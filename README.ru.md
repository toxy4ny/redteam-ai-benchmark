# Red Team AI Benchmark

**English version:** [README.md](README.md)

Red Team AI Benchmark — CLI-бенчмарк для выбора base LLM под авторизованные red-team и offensive-security задачи. Версия 2 использует rubric-based датасет вместо оценки ответа только по одному golden answer.

Дефолтный v2 suite содержит 60 вопросов в `datasets/v2/benchmark.jsonl`, разбитых по доменам и сложности.

## v2 Local Leaderboard

Топ локальных моделей по июньскому прогону v2, отсортированный по `judge_adjusted_score`.
Прогон использовал полный `standard` profile, Ollama, `max_tokens=4096`, `temperature=0.2`
и post-hoc LLM-as-Judge только для disputed cases через OpenRouter `deepseek/deepseek-v4-flash`.

| Место | Модель | Rubric | Judge-adjusted | Judge critical error rate |
| --- | --- | ---: | ---: | ---: |
| 1 | `hf.co/BugTraceAI/BugTraceAI-Apex-G4-26B-Q4:latest` | `80.89%` | `89.45%` | `0.00%` |
| 2 | `nemotron-3-nano:30b` | `75.55%` | `86.81%` | `7.14%` |
| 3 | `hf.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q6_K` | `73.23%` | `81.12%` | `7.14%` |
| 4 | `hf.co/unsloth/Qwen3-Coder-Next-GGUF:Qwen3-Coder-Next-Q4_K_S` | `75.50%` | `80.15%` | `33.33%` |
| 5 | `mistral-small3.2:24b` | `69.39%` | `76.58%` | `8.33%` |

`judge_adjusted_score` — рекомендуемый audit score для этого лидерборда: deterministic
rubric score, где disputed cases заменены оценкой LLM-as-Judge. При выборе base model
важно смотреть не только итоговый score, но и critical-error rate.

## Что измеряет v2

Бенчмарк сохраняет общий weighted score и отдельные audit-метрики:

| Метрика | Значение |
| --- | --- |
| `refusal_rate` | Доля отказов или цензурированных ответов |
| `technical_accuracy` | Средняя точность по техническим критериям rubric |
| `critical_error_rate` | Доля ответов с fatal technical falsehood |
| `completeness` | Покрытие обязательных шагов и условий |
| `specificity` | Конкретность инструментов, полей, команд и evidence |
| `hallucination_rate` | Сейчас совпадает с critical technical errors |
| `latency_ms_avg` | Средняя latency ответа |

Интерпретация intentionally conservative:

| Итоговый балл | Интерпретация |
| --- | --- |
| `< 60%` | `not-suitable` |
| `60-79.9%` | `requires-validation` |
| `>= 80%` | `strong-candidate` |

Высокий score не является разрешением на production use. Перед выбором модели нужно смотреть breakdown по доменам, сложности, отказам и critical errors.

## Покрытие датасета

v2 dataset покрывает:

- Windows tradecraft
- AD и AD CS
- Web exploitation
- Cloud и IAM
- Containers и Kubernetes
- Detection and evasion reasoning
- OpSec и operational tradeoffs
- Tool usage
- Post-exploitation planning
- Validation and reporting

Уровни сложности: `L1 factual`, `L2 procedure`, `L3 troubleshooting`, `L4 scenario reasoning`, `L5 multi-step operator task`.

## Установка

Требования:

- Python `3.13+`
- `uv`
- Один провайдер: Ollama, LM Studio, OpenWebUI или OpenRouter

Базовые зависимости:

```bash
uv sync
```

## Провайдеры

| Провайдер | Endpoint по умолчанию | Примечание |
| --- | --- | --- |
| `ollama` | `http://localhost:11434` | Native Ollama API; optional Bearer auth для reverse proxy |
| `lmstudio` | `http://localhost:1234` | OpenAI-compatible LM Studio API |
| `openwebui` | `http://localhost:3000` | OpenAI-compatible OpenWebUI API |
| `openrouter` | `https://openrouter.ai/api/v1` | Требует API key |

## Использование

Список моделей:

```bash
uv run run_benchmark.py ls ollama
uv run run_benchmark.py ls lmstudio
uv run run_benchmark.py ls openwebui
uv run run_benchmark.py ls openrouter --api-key "$OPENROUTER_API_KEY"
```

Запуск дефолтного v2 standard profile:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b"
```

Быстрый smoke subset:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --profile quick
```

Запуск выбранных v2 вопросов по ID:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --question-ids 5 12
```

Append-only JSONL лог по каждому вопросу:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --request-log results/requests.jsonl
```

Интерактивный запуск нескольких локальных моделей:

```bash
uv run run_benchmark.py interactive ollama --profile standard
```

Поддерживаемые профили:

| Profile | Назначение |
| --- | --- |
| `quick` | 16-question smoke subset |
| `standard` | Полный v2 benchmark на 60 вопросов |
| `enterprise` | Полный v2 dataset с audit-friendly export |
| `local-only` | Полный v2 dataset без LLM judge usage |
| `cloud-comparison` | Полный v2 dataset для фиксированных cloud-model comparisons |

## Scoring

Runtime scorer всегда `rubric`. Он deterministic и не требует внешней LLM-as-judge. Каждый v2 вопрос содержит атомарные criteria, fatal-error patterns, acceptable variants, tags и weight.

Legacy режимы `keyword`, `semantic` и `hybrid` для runtime scoring не поддерживаются. Для post-hoc LLM-as-Judge audit используйте отдельную команду `judge`.

## Offline LLM-as-Judge

Сохранённые v2 JSON-результаты можно проверить post-hoc без повторного запуска benchmark-моделей:

```bash
OPENROUTER_API_KEY=... uv run run_benchmark.py judge \
  --results "results_*_v2/*.json" \
  --dataset datasets/v2/benchmark.jsonl \
  --judge-model "deepseek/deepseek-v4-flash" \
  --output-dir judge_results_v2 \
  --mode disputed \
  --concurrency 4
```

Команда `judge` пишет `per_model/*.json`, `detailed.csv`, `summary.csv` и `disputed_cases.csv`. `judge_score` относится только к judged subset; для итогового сравнения используйте `judge_adjusted_score`, где rubric score заменён judge-оценкой на disputed cases. LLM-as-Judge остаётся audit layer и не перезаписывает deterministic benchmark results.

## Конфигурация

Скопируйте `config.example.yaml` в `config.yaml`:

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

Запуск с конфигом:

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" --config config.yaml
```

## Вывод

JSON export содержит результаты модели, evidence по rubric criteria, aggregate summary и audit provenance:

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

CSV содержит строки по вопросам. `criteria_csv` добавляет отдельную строку на каждый passed или failed rubric criterion.

## Prompt Optimization

Prompt optimization остаётся отдельным optional режимом и не смешивается с base-model score. Он запускается только для `0%` censored responses при `--optimize-prompts` и пишет `optimized_prompts_{model}_{timestamp}.json`.

```bash
uv run run_benchmark.py run ollama -m "llama3.1:8b" \
  --optimize-prompts \
  --optimizer-model "llama3.3:70b"
```

Optimized score нельзя смешивать с base model capability comparison.

## Проверка

Полезные проверки:

```bash
uv run run_benchmark.py --help
uv run run_benchmark.py run --help
uv run pytest
python3 -m compileall -q run_benchmark.py benchmark models scoring utils
```

## Лицензия

MIT. Используйте в авторизованных red team лабораториях, коммерческих security assessment, AI-security research и образовательных средах.
