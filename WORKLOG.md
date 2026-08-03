# WORKLOG — Local Deep Research Pipeline on RTX 5090

**Date:** 2026-08-03
**Goal:** Identify the state-of-the-art open-weight LLM that runs on a single RTX 5090 (32GB VRAM), download it, and get the `open_deep_research` pipeline running fully local with it — no cloud APIs.
**Outcome:** Working end-to-end. Final validated report: `report-20260803-132045.md` (1h43m runtime, zero errors).

---

## 1. Environment survey

- **GPU:** RTX 5090, 32GB VRAM (~2GB consumed by the KDE desktop/Xorg)
- **Host:** Linux, 60GB RAM, 1.4TB free on `/d1`, no passwordless sudo
- **Ollama:** v0.22.1 installed as a systemd service (existing models: gemma3/gemma4-era, nothing current)
- **Pipeline:** `/d1/local-research/open_deep_research` — LangChain's Open Deep Research (langchain 1.3.9, langgraph, langchain-openai 1.1.14). Four configurable model roles (summarization, research, compression, final report), all requiring tool calling + structured outputs. Search backends supported out of the box: Tavily / OpenAI-native / Anthropic-native / none.
- **No API keys anywhere on the box** (checked env, dotfiles, other project `.env` files) → needed a keyless search backend.

## 2. Model research (what's SOTA for 32GB in Aug 2026)

Web research converged on three serious contenders:

| Model | Shape | Notes |
|---|---|---|
| **Qwen3.6 35B-A3B** | MoE, 35B total / 3B active | ~24GB at Q4_K_M, 262K native context, Apache 2.0, tools+thinking+vision. Community consensus "best overall" for 32GB. |
| Qwen3.6 27B | Dense | Higher benchmark scores (77.2% SWE-bench Verified) but ~3× slower decode — worse fit for an agentic pipeline making 100+ calls. |
| Poolside Laguna XS 2.1 | MoE, 33B/3B | July 2026 release, excellent but coding/terminal-specialized (70.9% SWE-bench), OpenMDW-1.1 license. |
| gpt-oss-20b | MoE, 21B/3.6B | Fastest, cleanest tool-call JSON, but 128K context and lowest reasoning scores; needs Harmony format. |

**Chosen: `qwen3.6:35b`** — best quality/speed balance for research + report writing.

## 3. Setup

1. **Downloaded** `qwen3.6:35b` (23GB, Q4_K_M) via `ollama pull`.
2. **Created `qwen3.6-odr`** — a derived model with `PARAMETER num_ctx 65536` baked in via Modelfile (no sudo → can't set server-wide env in the systemd unit; per-model Modelfile is the workaround). Loads at **28GB, 100% GPU resident**.
3. **Wired the pipeline through Ollama's OpenAI-compatible endpoint** rather than `langchain-ollama`: the graph passes `max_tokens`/`api_key` kwargs which `ChatOpenAI` accepts and `ChatOllama` doesn't. `.env`:
   - `OPENAI_API_KEY=ollama`, `OPENAI_BASE_URL=http://localhost:11434/v1`
   - all four `*_MODEL` roles = `openai:qwen3.6-odr`
   - `SEARCH_API=duckduckgo`, `ALLOW_CLARIFICATION=false`, `MAX_CONCURRENT_RESEARCH_UNITS=2`, `MAX_CONTENT_LENGTH=8000`
4. **Added a keyless DuckDuckGo search backend** (patch to the repo):
   - `configuration.py`: new `SearchAPI.DUCKDUCKGO` enum member + UI option
   - `utils.py`: `duckduckgo_search_async()` mimics the Tavily response shape (so the existing dedupe → summarize → cite machinery works unchanged), fetches page content with aiohttp, extracts text with BeautifulSoup/lxml
   - Used the **`ddgs`** package (added via `uv add ddgs`) — the locked `duckduckgo-search` 8.x gets instant `202 Ratelimit`; `ddgs` rotates across Brave/Startpage/Mojeek/Google/etc.
5. **Smoke-tested** tool calling + structured output through the exact `init_chat_model` path the graph uses: tool calls well-formed, structured output valid, ~5-7s per call warm.
6. **Headless runner** `run_research.py`: streams graph progress, writes the report to `/d1/local-research/report-<timestamp>.md`.

## 4. Debugging saga (runs 1-6)

The pipeline "worked" mechanically on the first run but produced a hallucinated report. Getting to a *grounded* report took five more runs and uncovered three real bugs:

### Run 1 — hallucinated report (6.5 min)
Report claimed Qwen3.6/Laguna/gpt-oss "do not exist" — written from the model's stale training priors. **All 102 page-summarization calls had timed out** (hardcoded 60s timeout), so raw page dumps flooded the researcher's context, evidence got truncated away, and citations had no URLs.

### Run 2 — event-loop freeze diagnosed
Raised the timeout to 300s; still every summarization timed out, the GPU sat idle, and Ollama began unloading the model mid-run. Root cause: **my page-fetch helper parsed arbitrarily large HTML synchronously on the asyncio event loop** (`response.text()` had no size cap; `BeautifulSoup(html.parser)` on multi-MB pages froze the loop for minutes, so pending HTTP requests were never even sent).
**Fix:** size-capped streaming download (2MB), parse via `asyncio.to_thread` with lxml.

### Run 3 — Ollama OpenAI-compat quirks
Also tried disabling thinking for summarization (`reasoning_effort="none"`) to speed it up. Two discoveries:
- With reasoning disabled, Ollama **stops enforcing `response_format: json_schema`** → structured output returns plain text → validation errors.
- Switching to `method="function_calling"` mostly worked but Ollama **ignores forced `tool_choice`** → frequent `None` results when the model answered in prose.
**Fix:** reverted — summarization keeps thinking ON with the default json_schema method (it was never the bottleneck; 4.7-7s per call solo).

### Run 4 — the real bottleneck: throughput splitting
With the loop fixed, timeouts *still* hit — all at the same instant, with zero completions in 10 minutes and the GPU busy. Controlled experiment proved it: **6.8s for a solo summarization; 55-70s *each* for 8 concurrent** — Ollama admits every request simultaneously and divides decode throughput evenly, so the pipeline's ~40-request summarization burst meant *nothing* finished inside any reasonable timeout.
**Fix:** client-side `asyncio.Semaphore(3)` bounding in-flight summarizations per search call (+ 600s timeout as headroom).

### Run 5 — clean, but killed
Ran perfectly for 30+ minutes (zero warnings), then the background task was killed by something outside the pipeline. Relaunched detached with `nohup`.

### Run 6 — success ✅
**1h43m, ~130 model calls, zero warnings/timeouts.** The researcher's search queries visibly evolved as it read results ("Qwen3.6 benchmark results tool calling BFCL 2026", "Laguna XS 2.1 INT4 quantization VRAM requirements GGUF"), every page summary succeeded, and the final report is grounded: correct specs for all model families, real primary-source URLs (Poolside blog, Qwen HF cards, OpenAI gpt-oss announcement, Berkeley Function-Calling Leaderboard). Its #1 recommendation for agentic deep research — Qwen3.6-35B-A3B — is the model that wrote it. Known roughness: a few template-artifact headers, some duplicate/typo'd entries in the long source list.

## 5. Files changed / created

| File | Change |
|---|---|
| `open_deep_research/.env` | Local model + search config (new) |
| `open_deep_research/run_research.py` | Headless runner (new) |
| `src/open_deep_research/configuration.py` | `DUCKDUCKGO` search option |
| `src/open_deep_research/utils.py` | DuckDuckGo backend; threaded size-capped page parsing; summarization semaphore(3); 600s timeout |
| `pyproject.toml` / `uv.lock` | `+ ddgs` |
| Ollama | `qwen3.6:35b` pulled; `qwen3.6-odr` (64K ctx) created |
| `/d1/local-research/report-20260803-132045.md` | The validated deep-research report |

## 6. How to run it

```bash
cd /d1/local-research/open_deep_research
.venv/bin/python run_research.py "your research question"
# → /d1/local-research/report-<timestamp>.md   (~1.5-2h per run)
```

## 7. Possible next steps

- **Tavily free tier** (1,000 searches/mo): add `TAVILY_API_KEY` to `.env`, set `SEARCH_API=tavily` — cleaner results than the DDG backend rotation.
- Try `qwen3.6:27b` (dense) as `FINAL_REPORT_MODEL` for higher-quality prose while keeping the MoE for research (note: model swapping costs VRAM/load time per switch).
- If passwordless sudo is ever enabled: set `OLLAMA_KV_CACHE_TYPE=q8_0` + `OLLAMA_NUM_PARALLEL=2` in the service unit for parallel summarization slots.
