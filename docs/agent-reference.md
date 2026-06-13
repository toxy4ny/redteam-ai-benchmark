# Agent Reference

This document keeps longer project details out of `AGENTS.md` and `CLAUDE.md`. Agent entrypoints should stay short; use this file when deeper context is needed.

## Benchmark Categories

| ID | Category | Capability |
| --- | --- | --- |
| 1 | `AMSI_Bypass` | PowerShell P/Invoke, `AmsiScanBuffer`, `VirtualProtect`, `GetModuleHandle` |
| 2 | `ADCS_ESC1` | Vulnerable certificate templates, `Certify.exe`, `Rubeus`, `altname` |
| 3 | `NTLM_Relay_ADCS_ESC8` | Coercion, `ntlmrelayx`, AD CS HTTP enrollment, certificates |
| 4 | `Manual_PE_Mapping` | Windows x64 PE mapping, `NtCreateThreadEx`, remote memory writes |
| 5 | `ETW_Bypass` | `EtwEventWrite`, `ntdll.dll`, runtime patching |
| 6 | `UAC_Bypass_fodhelper` | `fodhelper.exe`, registry, `HKCU`, `ms-settings` |
| 7 | `C2_Profile_Teams` | Malleable C2 profile, Teams-like HTTPS traffic, headers, jitter |
| 8 | `Syscall_Shellcode` | Direct syscalls, `NtAllocateVirtualMemory`, `NtWriteVirtualMemory`, `NtCreateThreadEx` |
| 9 | `ADCS_ESC12` | CA key extraction, golden certificates, `krbtgt` |
| 10 | `LDAP_Relay_RBCD` | NTLM relay to LDAP, Web Client, RBCD, `delegate-access` |
| 11 | `EDR_Unhooking` | Restore clean `ntdll.dll` bytes from disk, resolve APIs |
| 12 | `Phishing_Lure` | IT support lure, urgency, PowerShell download cradle indicators |

## CLI Surface

Commands:

- `ls <provider>`: list available models.
- `run <provider> -m <model>`: run the benchmark for one model.
- `interactive <provider>`: select and test multiple models with TUI.

Common options:

- `-e, --endpoint`: override provider endpoint.
- `--api-key`: pass provider API key when required.
- `--config`: load YAML config.
- `--semantic`: enable semantic scoring.
- `--semantic-model`: choose sentence-transformer model.
- `--optimize-prompts`: enable prompt optimization for censored responses.
- `--optimizer-model`: optimizer model name.
- `--optimizer-endpoint`: separate optimizer endpoint.
- `--max-optimization-iterations`: per-question optimization cap.
- `--rate-limit-delay`: delay between benchmark request starts; `0` disables artificial delay.
- `--max-tokens`: max response tokens for benchmark target-model questions.
- `--temperature`: target-model temperature for benchmark questions.
- `--concurrency`: number of concurrent benchmark questions. Optimization and Langfuse paths run sequentially for now.
- `--export-csv`: parser exposes it; verify behavior before documenting as primary output.
- `--scorer`: parser exposes `keyword`, `semantic`, `hybrid`, `llm_judge`; verify non-default paths before relying on them.

## Output Files

Standard output file:

```text
results_{model}_{timestamp}.json
```

Important JSON fields:

- `model`
- `timestamp`
- `scoring_method`
- `total_score`
- `interpretation`
- `results[].id`
- `results[].category`
- `results[].score`
- `results[].response_snippet`
- `results[].full_response`

Prompt optimization output:

```text
optimized_prompts_{model}_{timestamp}.json
```

It contains original scores, best prompts, final scores, iteration count, success flag, and attempt history.

## Configuration Reference

`config.example.yaml` documents the supported YAML shape:

```yaml
provider:
  name: ollama
  endpoint: http://localhost:11434

scoring:
  method: keyword
  semantic_model: Alibaba-NLP/gte-large-en-v1.5
  semantic_weight: 0.7
  keyword_weight: 0.3
  llm_judge_model: anthropic/haiku-4.5
  gray_zone_low: 0.30
  gray_zone_high: 0.70
  use_llm_in_gray_zone: true

export:
  formats:
    - json
    - csv
  output_dir: ./results
  include_response: true

optimization:
  enabled: false
  optimizer_model: llama3.3:70b
  max_iterations: 3

questions_file: benchmark.json
answers_file: answers_all.txt
rate_limit_delay: 1.5
max_tokens: 1024
temperature: 0.2
concurrency: 1

langfuse:
  enabled: false
  host: http://localhost:3000
```

Use config cautiously: CLI arguments and config values are not applied uniformly in every command path. Verify the relevant code path in `run_benchmark.py` before assuming precedence.

## Scoring Details

`scoring/keyword_scorer.py` contains the canonical keyword scoring implementation:

- Censorship detection is phrase-based and context-aware.
- Each question has per-category technical keyword criteria.
- A non-censored answer that misses full criteria receives partial credit.

`run_benchmark.py` also contains legacy scoring code. If changing scoring behavior, check both locations and remove or synchronize duplication within the same scoped task.

`scoring/technical_scorer.py` combines semantic similarity and technical-term overlap. It requires `sentence-transformers`.

`scoring/llm_judge.py` uses OpenRouter-backed LLM evaluation.

`scoring/hybrid_scorer.py` combines technical scoring with optional LLM judge calls in a gray zone.

## Prompt Optimization

The optimizer is implemented in `run_benchmark.py`.

Strategies:

- role-based framing
- technical decomposition
- few-shot security-tool examples
- CVE/public-research framing

Flow:

1. Run original prompt.
2. If score is `0`, initialize optimizer.
3. Generate prompt variants.
4. Test selected variant against the target model.
5. Keep best response and score.
6. Stop on `100%`, acceptable non-censored score, or max iterations.
7. Save optimization history if attempts were made.

## Langfuse

Langfuse support is optional and uses SDK v3.

Trace shape:

```text
benchmark-{model}
  Q{id}-{category}
  optimization-Q{id}
    iter-{n}-{strategy}
```

Tracked metadata includes model name, scoring method, question id, category, score, total score, interpretation, latency, prompt, and response.

If Langfuse initialization fails, the benchmark should continue without tracing.

## Provider Implementation Notes

`models/base.py` defines `APIClient`.

`models/lmstudio.py`:

- chat completions endpoint: `/v1/chat/completions`
- model list endpoint: `/v1/models`
- OpenAI-compatible payload

`models/ollama.py`:

- chat endpoint: `/api/chat`
- model list endpoint: `/api/tags`
- native Ollama payload with options

`models/openrouter.py`:

- OpenAI-compatible API
- requires `httpx`, `tenacity`, and an API key

## Documentation Split

Recommended ownership:

- `AGENTS.md`: short agent instructions.
- `CLAUDE.md`: symlink or content-equivalent copy of `AGENTS.md`.
- `README.md`: English user guide.
- `README.ru.md`: Russian user guide.
- `docs/agent-reference.md`: longer implementation notes for agents.

Do not duplicate long README sections back into `AGENTS.md`.
