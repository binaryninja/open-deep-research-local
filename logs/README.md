# Run logs — the debugging trail

Raw logs from the six pipeline runs on the RTX 5090, 2026-08-03. They document every failure mode described in [WORKLOG.md](../WORKLOG.md) with timestamps you can verify. All model calls go to `localhost:11434` (Ollama); `response:` lines are the `ddgs` search-backend rotation.

| Log | Outcome | What to look for |
|---|---|---|
| `run1.log` | ⚠️ Completed but hallucinated ([the report](../examples/failure-case-run1-hallucinated-report.md)) | 102× `Summarization timed out after 60 seconds` — every page summary failed, so the model wrote the report from its training priors |
| `run2.log` | ❌ Killed after diagnosis | Timeout raised to 300s; note the total silence from `10:39:34` to `10:44:33` — the asyncio event loop was frozen by synchronous HTML parsing, so pending requests were never sent (all 23 timeouts fire in two same-millisecond batches) |
| `run3.log` | ❌ Killed after diagnosis | With thinking disabled, `'NoneType' object has no attribute 'summary'` every few seconds — Ollama's OpenAI-compat ignores forced `tool_choice`, so the model skipped the structured-output function call |
| `run4.log` | ❌ Killed after diagnosis | Event loop fixed, yet zero completions in 10 min with a busy GPU — Ollama splits decode throughput across all ~40 admitted requests, so none finish (measured separately: 6.8s solo → 55-70s each at 8 concurrent) |
| `run5.log` | ❌ External kill | The semaphore fix working: 30+ min of clean throughput, zero warnings — then the background task was killed by the environment (not the pipeline) |
| `run6.log` | ✅ **Success** | 1h43m, ~130 model calls, zero warnings. Watch the search queries evolve as the researcher reads results; report lands at `13:20:45` |

Fixes between runs (all in `src/open_deep_research/utils.py`):

- **run1 → run2:** summarization timeout 60s → 300s; page content cap 25K → 8K chars; model context 48K → 64K
- **run2 → run3:** size-capped (2MB) streaming fetch + HTML parsing moved off the event loop (`asyncio.to_thread` + lxml); tried `reasoning_effort="none"` + `method="function_calling"` for fast summaries
- **run3 → run4:** reverted the no-thinking optimization (Ollama drops `json_schema` enforcement without reasoning); timeout → 600s
- **run4 → run5:** `asyncio.Semaphore(3)` bounding in-flight summarization calls — the actual fix
- **run5 → run6:** same code, relaunched detached (`nohup`)
