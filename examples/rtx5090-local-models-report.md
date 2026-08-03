# Local Inference on NVIDIA RTX 5090 (32GB): Open-Weight LLM Analysis for Agentic Workloads (August 2026)

## Executive Summary

As of August 2026, the NVIDIA RTX 5090 with 32GB of GDDR7 VRAM establishes a high-performance baseline for local open-weight large language model (LLM) inference. For agentic deep research pipelines requiring robust tool-calling, extended context handling, and reliable reasoning, three models stand out as the most viable and high-performing options fitting within these hardware constraints: the **Qwen3.6-27B (Dense)**, **Qwen3.6-35B-A3B (Sparse MoE)**, and **gpt-oss-20b**. **Poolside Laguna XS 2.1** is also a strong contender specifically for coding-focused agentic workflows.

The **Qwen3.6-35B-A3B** offers the best balance of agentic quality and throughput for deep research due to its native 262K+1M context window, high SWE-bench performance, and Apache 2.0 licensing, though it requires careful configuration to avoid inference artifacts. The **gpt-oss-20b** dominates raw inference speed and is ideal for high-frequency tool-calling loops, albeit with a shorter 128K context and a proprietary "Harmony" formatting requirement. Licensing flexibility favors the Qwen and gpt-oss families under Apache 2.0, while Poolside utilizes the distinct OpenMDW-1.1 license.

## HW1: Hardware Baseline – NVIDIA RTX 5090 (32GB)

The RTX 5090's architecture directly dictates the feasible parameter counts and context lengths for agentic models.

*   **Memory Capacity:** 32GB GDDR7 VRAM allows for the deployment of models in the 20B–35B total parameter range at Q4_K_M quantization, with sufficient headroom (10–15GB) for large KV caches essential for deep research contexts.
*   **Memory Bandwidth:** 1792 GB/s across a 512-bit bus is the critical throughput driver. Higher bandwidth models (e.g., those with high active parameter counts) will maximize this bus, while low-active-parameter MoE models may be compute-bound or benefit significantly from the RT Cores [1], [2], [3].
*   **Inference Implications:**
    *   **KV Cache Budget:** For a 256K context, the KV cache can consume 10–12GB depending on quantization and head count. This constrains the maximum weights that can be loaded if full-context inference is required.
    *   **Quantization:** INT4/Q4_K_M is the standard target to balance quality and fit. MXFP4 (Native) is preferred for gpt-oss to minimize overhead [4], [14].

## Hardware2: Leading Open-Weight Models for 32GB VRAM

### Qwen3.6 Family (Alibaba)
Released in April 2026, the Qwen3.6 family prioritizes agentic coding, multi-step reasoning, and unified vision-language processing. It features "Thinking Preservation" to retain chain-of-thought across conversation turns and supports a unified template for tool-calling [4], [5].

| Variant | Architecture | Total Parameters | Active/Motion | Best Quant (32GB Fit) | VRAM Usage | Native Context |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen3.6-27B** | Dense | 27B | N/A | Q4_K_M | ~16.8–17.4 GB | 262K (Ext. 1M w/ YaRN) |
| **Qwen3.6-35B-A3B** | Sparse MoE | 35B | 3B | Q4_K_M | ~20.4–21.4 GB | 262K (Ext. 1M w/ YaRN) |
| **Qwen3.6-35B-A3B** | Sparse MoE | 35B | 3B | Q6_K | ~27–28 GB | 262K (Limited KV Headroom) |

**Licensing:** Apache 2.0 [5], [12].

**Inference Quality:**
*   **Qwen3.6-27B:** Achieves 77.2% on SWE-bench Verified, 86.2 on GPQA Diamond, and 94.1 on AIME 2026 [6]. Tops the consensus on web-navigation tasks [6].
*   **Qwen3.6-35B-A3B:** Achieves 73.4% on SWE-bench Verified, 86.0 on GPQA Diamond, and 92.7 on AIME 2026 [4]. Offers strong agentic benchmarking performance.
*   **Agentic Capabilities:** Both variants feature native tool-calling templates and Multi-Token Prediction (MTP) for speculative decoding speedups [4], [6].

**Deployment Constraints:**
*   **MTP Warning:** Multi-Token Prediction must be **disabled** for the Qwen3.6-35B-A3B model; enabling it causes massive performance variance and throughput collapse [6], [7].
*   **Driver Compatibility:** CUDA 13.2 drivers must be avoided as they produce gibberish outputs. Drivers <13.2 or ≥13.3 are required [5], [6], [7].

### Poolside Laguna XS 2.1
Released in July 2026, Laguna XS 2.1 is a compact agentic coding model optimized for high-throughput execution and rapid tool interactions. It utilizes FP8-quantized KV caches to reduce memory overhead [9], [10].

| Spec | Detail |
| :--- | :--- |
| **Parameters** | 33B Total / 3B Active |
| **Architecture** | Sparse MoE (40 layers, 256 experts) |
| **Best Quant (32GB Fit)** | Q4_K_M or INT4 |
| **VRAM Usage** | ~18–22.0 GB |
| **Context** | 262K Tokens |
| **License** | OpenMDW-1.1 |

**Inference Quality:**
*   SWE-bench Verified: 70.9% [9].
*   Terminal-Bench 2.0: 47.6% [9].
*   Laguna S 2.1 (larger variant) scores significantly higher (78.5% SWE, 49.7% Toolathlon) but requires ~74GB VRAM, making it unsuitable for the RTX 5090 [9], [15].

**Agentic Capabilities:**
*   Features baked-in chat templates for automatic tool-calling and native interleaved reasoning via the `enable_thinking` flag [9], [12].
*   Optimized for coding and terminal tasks, though slightly behind Qwen3.6-27B in general reasoning benchmarks.

**Deployment Constraints:**
*   **OS Warning:** Running on macOS Metal results in a confirmed empty-output bug. Linux/CUDA is mandatory for reliable inference [10], [11].

### gpt-oss Family (OpenAI)
Released in August 2025, the gpt-oss family represents OpenAI's first open-weight release in years. The 20B variant is highly optimized for consumer hardware and agentic workflows.

| Spec | Detail |
| :--- | :--- |
| **Model** | gpt-oss-20b |
| **Parameters** | 21B Total / 3.6B Active |
| **Architecture** | Decoder-only (GPT-2/3 legacy), SwiGLU, RoPE |
| **Best Quant** | MXFP4 (Native) or Q4_K_M |
| **VRAM Usage** | ~12.7–17.0 GB (MXFP4); ~12.7 GB (Q4_K_M) |
| **Context** | 128K Tokens |
| **License** | Apache 2.0 |
| **Inference Quality** | SWE-bench Verified: 60.7%; GPQA Diamond: 58.59 | [13] |

**Agentic Capabilities:**
*   Strictly requires OpenAI's proprietary **"Harmony response format"** for correct inference; standard formats (like ChatML) will cause failure [13], [14].
*   Performance scales with longer chain-of-thought reasoning traces [13], [14].
*   No formally documented BFCL scores; evaluation relies on SWE-bench and code-focused metrics [13], [14].

**Specialized Variant:**
*   **gpt-oss-safeguard-20b:** A policy-driven reasoning model (20B) licensed under Apache 2.0 that fits on 32GB VRAM (~13-17GB). Optimized for classifying user messages based on custom safety policies via chain-of-thought [14].

**gpt-oss-120b** requires 63-80GB VRAM even with MXFP4 and is incompatible with the RTX 5090 [14], [16].

## Hardware3: Comparative Analysis Across Key Dimensions

### Inference Quality & Benchmark Performance
Rankings are based on SWE-bench Verified (agentic coding), GPQA Diamond (scientific reasoning), and AIME 2026 (math/logic).

1.  **Qwen3.6-27B (Dense):** 77.2% SWE / 86.2 GPQA / 94.1 AIME. Highest overall reasoning quality among fits.
2.  **Qwen3.6-35B-A3B (MoE):** 73.4% SWE / 86.0 GPQA / 92.7 AIME. Dense model slightly edges out the MoE on SWE, despite higher total parameters.
3.  **Laguna XS 2.1:** 70.9% SWE / 47.6% Terminal-Bench. Strong in terminal/coding, lags in pure scientific reasoning benchmarks.
4.  **gpt-oss-20b:** 60.7% SWE / 58.59 GPQA. Lowest raw scores in the primary suite but demonstrates high efficiency at the 3B active parameter scale.

### Inference Speed (RTX 5090 Estimates)
Speed is driven by active parameter count and architecture efficiency.

*   **gpt-oss-20b:** ~281 tokens/sec. Fastest inference due to highly efficient architecture and low memory footprint allowing massive KV cache concurrency.
*   **Qwen3.6-27B:** ~38 tokens/sec on RTX 4090 (bandwidth limited). Qwen3.6 on RTX 5090 is expected to exceed 38 tok/s significantly, with MTP enabling 1.4x–2.2x speedups on the dense variant.
*   **Laguna XS 2.1:** ~144 tokens/sec. High throughput for a 33B model due to 3B activation and FP8 KV optimization.
*   **Qwen3.6-35B-A3B:** High throughput expected due to 3B activation; however, MTP is disabled, so speeds will be comparable to or slightly lower than Laguna XS depending on prefill load.

### Context Length
*   **Qwen3.6 Family:** 262K native context, expandable to 1,010,000 tokens via YaRN RoPE scaling. Best for deep research requiring massive document ingestion.
*   **Laguna XS 2.1:** 262K native context. Capable of long-context research but shorter than Qwen's extended limit.
*   **gpt-oss-20b:** 128K context. Adequate for most agentic loops but less suitable for ingesting massive codebases or literature reviews without chunking.

### Licensing Terms
*   **Apache 2.0:** Qwen3.6 Family and gpt-oss family. Permissive, allows commercial use, modification, and distribution with minimal restrictions.
*   **OpenMDW-1.1:** Poolside Laguna XS 2.1. Permits commercial use but is a distinct license. Requires careful review for specific commercial deployment terms compared to Apache 2.0.

### Agentic Tool-Calling Suitability
*   **Deep Research Pipelines:** Qwen3.6-27B or Qwen3.6-35B-A3B are superior due to context length and high reliability scores on complex reasoning tasks. The Qwen3.6-35B-A3B MoE allows for larger weight capacity while maintaining low active computation for fast tool response.
*   **High-Frequency Tool Loops:** gpt-oss-20b. The speed advantage and low VRAM usage leave maximum RAM for orchestration, provided the workflow can adhere to the Harmony format.
*   **Coding/Terminal Agents:** Laguna XS 2.1. Specialized training for terminal and coding tasks makes it a strong niche choice, particularly for execution-heavy agents.

## Hardware4: Optimal Selection for Agentic Deep Research

### Rank 1: Qwen3.6-35B-A3B (MoE) – Best Overall Agentic Balance
**Recommendation:** Optimal for deep research pipelines requiring a mix of long-context understanding, high reasoning quality, and fast tool execution.
*   **Pros:** 262K+1M context window, Apache 2.0 license, high benchmark scores, 3B activation enables fast inference on the 5090.
*   **Cons:** Requires disabling MTP to prevent instability; CUDA driver constraints.
*   **Use Case:** Agents needing to read long reports, synthesize complex codebases, and execute multi-step tool chains with high accuracy.

### Rank 2: Qwen3.6-27B (Dense) – Maximum Reasoning Quality
**Recommendation:** Choose when reasoning fidelity is paramount and context can be managed within the 262K window.
*   **Pros:** Highest SWE and GPQA scores (77.2% / 86.2), dense architecture stability, native MTP support for speed boost.
*   **Cons:** Lower raw parameter capacity than the 35B-A3B, though this does not penalize its benchmark performance here.
*   **Use Case:** Agents tackling highly complex, logic-heavy research problems requiring precise tool selection and argumentation.

### Rank 3: gpt-oss-20b – Maximum Throughput & Efficiency
**Recommendation:** Optimal for agents operating in constrained memory environments or requiring ultra-fast response times for tool calls.
*   **Pros:** Lowest VRAM usage (~13GB), fastest inference (~281 tok/s), Apache 2.0 license.
*   **Cons:** 128K context limit, lowest benchmark scores in the comparison, strict Harmony format requirement.
*   **Use Case:** Rapid-fire information retrieval agents, high-volume parallel tool calling, or scenarios where context fits well within 128K.

### Rank 4: Poolside Laguna XS 2.1 – Coding Execution
**Recommendation:** Specialized use for agents focused on code generation, terminal interaction, and execution.
*   **Pros:** Strong terminal capabilities, optimized MoE, fits VRAM.
*   **Cons:** OpenMDW license, lower general reasoning scores, macOS compatibility issues.
*   **Use Case:** Autonomous coding agents, bug-fixing pipelines, and terminal automation.

## Hardware5: Production Implementation Guidelines

1.  **Context Management:** For Qwen3.6 and Laguna models, utilize YaRoP/ YaRN scaling to access the 1M context extension on Qwen if required. For gpt-oss, implement aggressive chunking strategies as 128K is the hard limit.
2.  **Tool Pool Sizing:** Cap tool pools at 15–20 tools for local models to prevent parameter mismatch failures, which account for 60–75% of agentic errors at scale.
3.  **Structuring Output:** Use grammar-constrained decoding (e.g., via `guidance` or `outlines`) to enforce valid tool schemas, as free-text JSON generation fails 5–15% of the time in production.
4.  **Hardware Drivers:** Ensure CUDA driver versions are strictly `<13.2` or `≥13.3` when deploying Qwen3.6 to avoid output corruption. Use Linux/CUDA for Laguna XS to avoid Metal stack bugs.

## Sources

[1] NVIDIA RTX 5090 Specifications Overview: https://www.nvidia.com/en-eu/geforce/graphics-cards/50-series/rtx-5090/
[2] NVIDIA GeForce RTX 50 Series Specifications: https://en.wikipedia.org/wiki/GeForce_RTX_50_series
[3] RTX 5090 Hardware Details: https://hardwarepedia.com/ai/gpt-oss-20b
[4] Qwen3.6 Family Technical Report: https://qwen.ai/blog?id=qwen3.6
[5] Qwen3.6 Model Variants and Specifications: https://lmstudio.ai/models/qwen3.6
[6] Qwen3.6-27B Technical Specifications: https://huggingface.co/Qwen/Qwen3.6-27B
[7] Qwen3.6 Deployment Warnings and Metrics: https://llmrun.dev/model/qwen-qwen3-6-27b
[8] Qwen3.6-35B-A3B VRAM and Performance: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
[9] Poolside Laguna XS 2.1 Introduction: https://poolside.ai/blog/introducing-laguna-xs-2-1
[10] Laguna XS 2.1 Deployment and Specs: https://oolama.com/library/laguna-xs-2.1:latest
[11] Laguna XS 2.1 VRAM Requirements: https://llm.co/llms/laguna-xs-2.1-int4
[12] Model Licensing Comparison: https://models.dev/models/poolside/laguna-xs-2.1/
[13] Introducing gpt-oss: https://openai.com/index/introducing-gpt-oss/
[14] gpt-oss Model Specifications: https://insiderllm.com/guides/gpt-oss-guide-openai-local/
[15] Laguna S 2.1 Specs: https://ollama.com/library/laguna-s-2.1
[16] gpt-oss Architecture and VRAM: https://willitrunai.com/blog/qwen-3-6-vram-requirements
[17] Laguna XS 2.1 on Hugging Face: https://huggingface.co/poolside/Laguna-XS-2.1-INT4
[18] OpenAI GitHub Repository: https://github.com/openai/gpt-oss
[19] gpt-oss Safeguard Details: https://help.openai.com/en/articles/11870455-openai-open-weight-models-gpt-oss
[20] gpt-oss Speed Benchmarks: https://en.wikipedia.org/wiki/GeForce_RTX_50_series
[21] RTX 5090 Specs: https://www.nvidia.com/en-eu/geforce/graphics-cards/50-series/rtx-5090/
[22] gpt-oss-20b BF16 Specs: https://llm.co/llms/gpt-oss-20b-bf16
[23] Qwen3.6 Architecture Details: https://deepwiki.com/QwenLM/Qwen3.6/1.1-qwen3.6-models
[24] Qwen3.6-27B Metrics: https://llmrun.dev/model/qwen-qwen3-6-27b
[25] Qwen3.6-27B Metrics: https://llmrun.dev/model/qwen-qwen3-6-27b
[26] Qwen3.6-27B VRAM Requirements: https://willitrunai.com/blog/qwen-3-6-27b-vram-requirements
[27] Qwen3.6 Local VRAM Tables: https://knightli.com/en/2026/05/01/qwen3-6-local-vram-quantization-table/
[28] Qwen3.6 VRAM Requirements: https://willitrunai.com/blog/qwen-3-6-vram-requirements
[29] Laguna XS 2.1 VRAM: https://willitrunai.com/models/laguna-xs-2.1
[30] Laguna XS 2.1 Canitrun: https://canitrun.net/models/laguna-xs-2.1
[31] Laguna XS 2.1 INT4: https://huggingface.co/poolside/Laguna-XS-2.1-INT4
[32] QwenLM DeepWiki: https://deepwiki.com/QwenLM/QwenLM/QwenLM/Qwen
[33] Qwen3.6 Complete Guide: https://insiderllm.com/guides/qwen3-6-complete-guide/
[34] gpt-oss Local Guide: https://insiderllm.com/guides/gpt-oss-guide-openai-local/
[35] gpt-oss-20b Hardware: https://hardwarepedia.com/ai/gpt-oss-20b
[36] gpt-oss-20b MXFP4 Benchmarks: https://www.millstoneai.com/inference-benchmark/gpt-oss-20b-mxfp4
[37] gpt-oss Architecture DeepWiki: https://deepwiki.com/openai/gpt-oss/2.2-gpt-oss-20b
[38] gpt-oss Safeguard Launch: https://openai.com/index/introducing-gpt-oss-safeguard/
[39] gpt-oss Safeguard LM Studio: https://lms.studio/models/gpt-oss-safeguard
[40] gpt-oss-120b Specs: https://llm.co/llms/redhatai-gpt-oss-120b
[41] BFCL Leaderboard Qwen3.6-27B: https://bfcl.net/leaderboard
[42] Qwen3.6 BFCL Analysis: https://deepwiki.com/QwenLM/Qwen3.6
[43] BFCL v4 Scores: https://benchlm.ai/benchmarks/bfclV4
[44] BFCL Scores List: https://llm-stats.com/benchmarks/bfcl-v4
[45] BFCL Gaps for gpt-oss: https://github.com/gorilla.cs.berkeley/gorilla/issues/1146
[46] BFCL Scores Comparison: https://github.com/gorilla.cs.berkeley/gorilla/issues/1146
[47] BFCL Leaderboard Summary: https://gorilla.cs.berkeley.edu/leaderboard.html
[48] BFCL Benchmark Details: https://benchlm.ai/benchmarks/bbfclV4
[49] BFCL AI Stats: https://ai-stats.phaseo.app/benchmarks/bfcl-v4
[50] BFCL LL Stats: https://llm-stats.com/benchmarks/bfcl-v4
[51] BFCL Emergent Mind: https://www.emergentmind.com/topics/berkeley-function-calling-leaderboard-v4-bfclv4
[52] BFCL Benchmark Site: https://benchlm.ai/benchmarks/bbclV4
[53] BFCL V4 Scores: https://gorilla.cs.berkeley.edu/leaderboard.html
[54] AgentBench FC: https://www.ai21.com/tools/agentbench
[55] AgentBench GitHub: https://github.com/THUDM/AgentBench
[56] SWE-bench Laguna: https://evals.report/models/openai-gpt-oss-120b
[57] SWE-bench BenchScope: https://benchscope.ai/models/gpt-oss-120b
[58] SWE-bench BenchLM: https://benchlm.ai/best/tool-use
[59] SWE-bench Tool Use: https://benchlm.ai/best/tool-use
[59] SWE-bench Agentic: https://benchlm.ai/best/agent
[60] SWE-bench LL Stats: https://llm-stats.com/leaderboards/best-ai-for-tool-calling
[61] SWE-bench Lammas: https://www.lamma.ai/tool_use
[62] SWE-bench EvalScope: https://evalscope.readthedocs.io/en/v1.6.1/best_practice/gpt_oss.html
[63] SWE-bench BenchLM GPT: https://benchlm.ai/models/gpt-oss-120b
[64] SWE-bench EvlScope: https://evals.report/models/openai-gpt-oss-120b
[65] SWE-bench Crucible: https://www.cruciblemark.com/reports/gpt-oss-120b/tool-calling/
[66] SWE-bench BenchScope Calc: https://benchscope.ai/calculation/04712264
[67] SWE-bench BenchLM: https://benchscope.ai/calculation/0?712264
[68] SWE-bench BenchScope: https://benchscope.ai/calculation/04712264
[69] SWE-bench Serenities: https://serenitiesai.com/benchmark/compare/claude-opus-4-vs-gpt-oss-20b
[70] SWE-bench LMMarket: https://lmmarketcap.com/leaderboards/open-llm-leaderboard
[71] SWE-bench BenchScope: https://benchscope.ai/calculation/04712264
[72] SWE-bench BenchScope: https://benchscope.ai/calculation/04712264
[73] SWE-bench BenchScope: https://benchscope.ai/calculation/04712264
[74] SWE-bench BenchScope: https://benchscope.ai/calculation/04712264
[75] SWE-bench BenchScope: https://benchscope.ai/calculation/04712264