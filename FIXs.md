# Рекомендации по оптимизации скорости

## Краткое резюме

Главный тормоз кодовой базы сейчас не CPU Python, а сетевой и LLM I/O: вопросы выполняются последовательно, после каждого вопроса стоит фиксированная задержка, HTTP-клиенты не переиспользуют соединения, а semantic scoring кодирует тексты по одному. Первые улучшения стоит делать вокруг сокращения лишних ожиданий и запросов, затем добавлять управляемую параллельность.

## P0: Использовать настройки скорости из config

Сейчас `config.example.yaml` содержит `rate_limit_delay`, `max_tokens` и `temperature`, но основной путь выполнения использует часть параметров не полностью.

Что изменить:

- заменить хардкод `time.sleep(1.5)` в `run_benchmark.py` на значение из config или CLI;
- передавать `max_tokens` из config/CLI в `client.query(...)`;
- добавить передачу `temperature` в provider clients, если настройка должна быть публичной;
- сохранить безопасные дефолты для обратной совместимости.

Ожидаемый эффект:

- для локальных Ollama/LM Studio можно убрать лишние примерно 16.5 секунд на прогон из 12 вопросов при `rate_limit_delay=0`;
- контроль длины ответа через `max_tokens` часто дает больший выигрыш, чем микроправки в Python.

Где смотреть:

- `run_benchmark.py`: основные циклы benchmark и interactive;
- `utils/config.py`: `BenchmarkConfig`;
- `config.example.yaml`: публичные настройки benchmark.

## P0: Переиспользовать HTTP-клиенты и соединения

Provider clients сейчас создают отдельные HTTP-запросы без долгоживущей сессии. Для OpenRouter `httpx.Client` создается внутри каждого запроса.

Что изменить:

- в `OllamaClient` и `LMStudioClient` завести `requests.Session`;
- в `OpenRouterClient` держать один `httpx.Client` на экземпляр;
- добавить `close()` или context manager для клиентов;
- не пересоздавать headers и client objects в горячем цикле без необходимости.

Ожидаемый эффект:

- меньше latency на TCP/TLS/HTTP setup;
- стабильнее поведение при серии запросов;
- особенно полезно для OpenRouter и удаленных endpoints.

Где смотреть:

- `models/ollama.py`;
- `models/lmstudio.py`;
- `models/openrouter.py`;
- `models/base.py`.

## P0: Не тестировать original prompt повторно в optimizer

Сейчас основной benchmark сначала получает response и score для исходного prompt. Если score равен `0`, `PromptOptimizer.optimize_prompt()` повторно тестирует тот же original prompt.

Что изменить:

- передавать в optimizer уже полученные `initial_response` и `initial_score`;
- начинать optimization history с этих данных;
- не делать второй identical target-model query для iteration 0.

Ожидаемый эффект:

- экономия одного LLM-вызова на каждый censored вопрос;
- заметный выигрыш при `--optimize-prompts`, где каждый вопрос и так может запускать несколько дополнительных запросов.

Где смотреть:

- `run_benchmark.py`: блоки `if score == 0 and optimizer`;
- `PromptOptimizer.optimize_prompt()`.

## P1: Добавить управляемую параллельность

Вопросы benchmark независимы, но сейчас выполняются строго последовательно.

Что изменить:

- добавить CLI/config параметр `concurrency` с дефолтом `1`;
- для удаленных OpenAI-compatible провайдеров использовать `ThreadPoolExecutor` или async `httpx`;
- сохранять порядок результатов по `question id`, даже если ответы приходят не по порядку;
- учитывать rate limit: задержка должна применяться к планировщику запросов, а не слепо после каждого completed task;
- для Ollama оставить conservative default `1`, потому что локальная модель часто упирается в один GPU.

Ожидаемый эффект:

- на OpenRouter и multi-worker endpoints wall-clock time может снизиться почти пропорционально `concurrency`, пока не упремся в rate limit;
- interactive multi-model режим можно ускорить за счет параллельного прогона моделей только если provider это выдерживает.

Где смотреть:

- `run_benchmark.py`: `cmd_run_benchmark`, `cmd_interactive`;
- provider clients в `models/`.

## P1: Batch-encoding для semantic scoring

Semantic scoring сейчас кодирует reference answers и model responses по одному. `sentence-transformers` эффективнее работает с batch input.

Что изменить:

- кодировать все reference answers одним вызовом `model.encode(list_of_answers, ...)`;
- хранить mapping `q_id -> embedding row`;
- для анализа готовых результатов кодировать responses батчами по файлу или по модели;
- вынести общую semantic scoring логику из `run_benchmark.py` и `analyze_semantic.py`.

Ожидаемый эффект:

- быстрее запуск semantic scoring;
- меньше overhead на Python calls и model invocation;
- особенно полезно для `analyze_semantic.py`, где несколько embedding-моделей применяются к множеству сохраненных results.

Где смотреть:

- `run_benchmark.py`: `SemanticScorer`;
- `scoring/technical_scorer.py`;
- `analyze_semantic.py`.

## P1: Реально подключить scorer factory

CLI принимает `--scorer {keyword,semantic,hybrid,llm_judge}`, но подтвержденный основной путь сейчас завязан на `--semantic`, а `score_response()` продублирован.

Что изменить:

- сделать единую фабрику scorer'ов;
- убрать дублирующий `score_response()` из `run_benchmark.py`;
- использовать `scoring/keyword_scorer.py` как один source of truth для keyword scoring;
- подключить `technical`, `hybrid` и `llm_judge` только после проверки CLI-пути тестами.

Ожидаемый эффект:

- меньше риска рассинхрона scoring-логики;
- проще оптимизировать scoring независимо от benchmark orchestration;
- проще добавлять быстрые и медленные scoring profiles.

Где смотреть:

- `run_benchmark.py`;
- `scoring/keyword_scorer.py`;
- `scoring/technical_scorer.py`;
- `scoring/hybrid_scorer.py`;
- `scoring/llm_judge.py`.

## P1: Убрать повторную загрузку вопросов в interactive

В interactive-режиме вопросы можно загрузить один раз перед циклом по выбранным моделям.

Что изменить:

- перенести `questions = load_questions()` перед `for model_name in selected_model_names`;
- использовать тот же список для каждого selected model.

Ожидаемый эффект:

- небольшой выигрыш по I/O;
- меньше дублирования и проще будущая параллелизация.

Где смотреть:

- `run_benchmark.py`: `cmd_interactive`.

## P2: Снизить overhead Langfuse в горячем цикле

При включенной трассировке spans создаются и обновляются на каждый вопрос. Это правильно функционально, но добавляет overhead в горячем пути.

Что изменить:

- буферизовать lightweight event data во время benchmark;
- создавать/flush spans пачкой в конце, если SDK и формат трассировки это позволяют;
- как минимум измерить overhead с Langfuse on/off перед изменениями.

Ожидаемый эффект:

- меньше latency при включенном observability;
- более предсказуемая производительность benchmark.

Где смотреть:

- `run_benchmark.py`: `LangfuseTracer`, вызовы `log_generation()` и `log_optimization_attempt()`.

## P2: Нормализовать output/export путь

В коде есть `utils/export.py`, CLI принимает `--export-csv`, но основной verified path сохраняет JSON через локальную функцию `save_results()`.

Что изменить:

- использовать `BenchmarkExporter` как единый output path;
- сделать `--output`, `--export-csv` и `config.export` рабочими в одном месте;
- не сериализовать лишние поля, если пользователь отключил full response export.

Ожидаемый эффект:

- меньше лишнего кода в `run_benchmark.py`;
- можно уменьшить размер results-файлов и I/O при массовых прогонах.

Где смотреть:

- `run_benchmark.py`;
- `utils/export.py`;
- `config.example.yaml`.

## Рекомендуемый порядок внедрения

1. Сделать config-driven `rate_limit_delay`, `max_tokens`, `temperature`.
2. Добавить session reuse в provider clients.
3. Убрать повторный original prompt query в optimizer.
4. Убрать дублирование keyword scoring и сделать scorer factory.
5. Добавить batch-encoding для semantic scoring.
6. Добавить `--concurrency` с дефолтом `1`.
7. Нормализовать export path и Langfuse overhead.

## Как проверять

- `uv run pytest`
- `python3 -m compileall -q run_benchmark.py models scoring utils`
- `uv run run_benchmark.py run --help`
- smoke test на локальном provider:
  - `uv run run_benchmark.py run ollama -m "<model>" --config config.yaml`
- для semantic path:
  - `uv sync --extra semantic`
  - `uv run run_benchmark.py run ollama -m "<model>" --semantic`
- для performance regression:
  - замерить wall-clock time до и после на одном и том же provider/model;
  - сравнить total score и порядок results;
  - отдельно замерить Langfuse on/off, если tracing включен.

## Критерии успешного обновления

- benchmark сохраняет тот же формат результатов или явно мигрирует его с документацией;
- порядок результатов стабилен по `question id`;
- дефолтное поведение без config не ломается;
- `rate_limit_delay=0` действительно убирает искусственную задержку;
- `max_tokens` реально влияет на provider payload;
- semantic scoring дает те же score buckets на существующих тестах;
- при `concurrency=1` поведение совпадает с последовательным режимом.

## Критерии неуспешного обновления

- результаты приходят в случайном порядке;
- retry/rate-limit логика начинает дублировать запросы без контроля;
- OpenRouter или локальные endpoints получают больше параллельных запросов, чем задано пользователем;
- config содержит параметры, которые не влияют на выполнение;
- scoring логика расходится между `run_benchmark.py` и `scoring/`;
- full benchmark нельзя воспроизвести теми же командами из README.

## Чего не делать

- Не переписывать весь CLI за один проход.
- Не менять benchmark questions или scoring criteria в рамках performance PR.
- Не включать параллельность по умолчанию больше `1` без явного решения.
- Не удалять Langfuse или prompt optimization ради ускорения.
- Не документировать `hybrid`/`llm_judge` как fully supported, пока CLI-путь не покрыт smoke-тестом.
- Не трогать unrelated файлы и существующие пользовательские изменения в dirty worktree.
