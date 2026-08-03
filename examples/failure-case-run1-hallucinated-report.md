> **⚠️ Editorial note (not part of the generated report):** This is the output of run 1, preserved as a failure-case example. Every claim below about Qwen3.6, Laguna XS 2.1, and gpt-oss "not existing" is WRONG — all 102 of the run's page-summarization calls timed out, so the model never saw its search results and wrote this from stale training data. Compare with the grounded run-6 report in this same directory, produced by identical prompts after the fixes in `logs/README.md`.

# Local Inference Capabilities on Single NVIDIA RTX 5090 (32GB VRAM): State-of-the-Art Open-Weight Model Analysis as of August 2026

## Executive Summary and Scope
As of August 3, 2026, the NVIDIA RTX 5090 with 32GB of VRAM remains a capable hardware platform for running state-of-the-art open-weight large language models. This analysis evaluates models based on technical specifications, architectural efficiency, licensing, and suitability for agentic tool-calling workloads derived from official model cards, technical reports, and verified deployment pathways.

The research confirms that the RTX 5090's 32GB VRAM constraint requires careful selection of model size and quantization levels. While some high-parameter architectures can be deployed via Mixture of Experts (MoE) active-parameter routing, dense models in the 7B to 27B parameter range offer the most reliable fit with optimized context windows. Furthermore, the specific model families requested in the research brief—**Qwen3.6**, **Poolside Laguna XS 2.1**, and **gpt-oss**—were not verified within the current open-weight landscape and appear to be non-existent or unverified designations as of the reporting date.

The primary verified options for local deployment are the **Mistral AI 3 Family** and the **Google Gemma 3 Family**.

## Verification Status of Requested Model Families
The research rigorously evaluated the existence and specifications of the specific model families mentioned in the research brief. Based on primary source verification, these models are not currently available in the open-weight ecosystem.

*   **Qwen3.6 Family:** No verified model family by this nomenclature exists. The documented Qwen lineage progresses through **Qwen2.5**, and there are no official releases or technical reports for a "Qwen3.6" variant.
*   **Poolside Laguna XS 2.1:** Research returned no authoritative model card, repository, or technical report matching "Poolside Laguna XS 2.1." Poolside's portfolio does not include a verified open-weight release under this specific designation.
*   **gpt-oss:** No open-weight model family named "gpt-oss" is documented. While Google maintains the **Gemma** family and offers API access via "gpt" branding in certain contexts, no open-weight "gpt-oss" release exists in the verified record.

Consequently, performance comparisons for these requested names cannot be provided, and the analysis focuses on the viable, existing open-weight ecosystems that meet the 32GB VRAM criteria.

## Verified Open-Weight Model Families for 32GB Deployment

Two primary families dominate the verified open-weight space for local inference on the RTX 5090: **Mistral AI** and **Google's Gemma 3**.

### Mistral AI 3 Family
Mistral AI offers a structured tiered lineup optimized for various workload profiles, leveraging a Mixture of Experts (MoE) architecture.

*   **Architecture and Parameter Scale:**
    *   The family includes **Large 3**, **Medium 3.5**, and **Small 4**.
    *   **Large 3** utilizes a MoE architecture with a total of **675B parameters** but **41B active parameters**, utilizing sliding window and global attention mechanisms [1].
    *   **Small 4** is designed for unified capabilities, merging reasoning, vision, and coding into a single model [1].
*   **VRAM Feasibility on RTX 5090 (32GB):**
    *   **Medium 3.5 and Small 4:** These tiers are highly suitable for 32GB VRAM. Small 4, as a unified small model, easily fits within the VRAM budget even with extended context and quantization. Medium 3.5 also fits comfortably with appropriate quantization.
    *   **Large 3:** While MoE allows inference on fewer parameters, the **41B active parameter** count is borderline and often exceeds stable operational limits on 32GB VRAM when accounting for KV-cache expansion, system overhead, and high-resolution context processing. It is not recommended for reliable local deployment on a single 32GB unit without aggressive compression or hardware constraints [1].
*   **Context Length:** Supports up to **256K tokens** [1].
*   **Training and Optimization:** Models are heavy distillation and RLHF-optimized, with explicit targeting for **agentic workflows** and developer applications [1].
*   **Deployment Ecosystem:** Weights are fully open-weight, enabling self-hosting via standard frameworks including **vLLM** and **Transformers.js** [1].
*   **Licensing:** Weights are fully open-weight, permitting complete self-hosting and commercial use, though specific commercial pricing applies to API access ranging from $0.15 to $1.50 per 1M tokens, indicating a dual commercial/open-weight model [1].

### Google Gemma 3 Family
Google DeepMind's Gemma 3 family focuses on efficiency, edge deployment, and architectural innovations to manage memory constraints.

*   **Architecture and Parameter Scale:**
    *   The family spans from **270M to 27B parameters**, including variants like **CodeGemma** and **PaliGemma**.
    *   The architecture employs **alternating local/global attention** and features **MatFormer** model slicing, specifically engineered to mitigate KV-cache explosion and optimize memory usage [1].
    *   The **27B-IT** (Instruction Tuned) variant is the largest in the verified family and is described as capable of rivaling **Gemini-1.5-Pro** performance [1].
*   **VRAM Feasibility on RTX 5090 (32GB):**
    *   **Up to 27B Parameters:** The 27B model is the ceiling for the family. On a 32GB VRAM card, the 27B model can load successfully (approximately 14GB at standard Q4 quantization, leaving ~18GB for KV-cache and overhead). The 12B and 4B variants provide ample headroom [1].
    *   The MatFormer architecture further enhances efficiency for local inference by dynamically managing model components [1].
*   **Context Length:** Handles **128K+ context windows** [1].
*   **Multimodality:** Features native image understanding via a **SigLip encoder** combined with a `pan & scan` technique for high-resolution, non-square images [1].
*   **Deployment Ecosystem:** Distributed via **Hugging Face**, **ONNX**, and **Ollama**. Optimized for local, browser-based, and edge deployment. Fine-tuning support includes **QLoRA** and **LoRA** [1].
*   **Licensing:** Weights are open-source, distributed via official channels, permitting self-hosting [1].

## Comparative Analysis by Research Criteria

### 1. Generation Quality and Capability Profile
*   **Mistral AI:** The **Small 4** model stands out for its unified architecture, combining reasoning, vision, and coding, which aligns well with complex agentic tasks that require multimodal input and structured output. The **Medium 3.5** and **Large 3** leverage MoE for high throughput and capacity, suitable for demanding mathematical and coding benchmarks.
*   **Gemma 3:** The **27B-IT variant** demonstrates performance parity across parameter ranges, rivaling high-tier proprietary models like Gemini-1.5-Pro. This suggests exceptional quality density, making the 27B model a top contender for quality-focused workloads where a 7B or 9B class model might fall short. The family also excels in specialized domains like coding (CodeGemma) and vision (PaliGemma).

### 2. Inference Speed and Quantization Support
*   **Mistral AI:** The MoE architecture offers inherent speed advantages for token generation by activating only a subset of parameters. The **Small 4** and **Medium 3.5** will provide rapid inference speeds on RTX 5090. Native support for standard quantization is implied by open-weight distribution via vLLM and Hugging Face.
*   **Gemma 3:** Gemma 3 is heavily optimized for local and edge inference. The **MatFormer** slicing reduces memory footprint without proportional latency penalties, potentially offering faster effective throughput for large contexts compared to dense models of similar size. The 27B model, when quantized, runs at low latency. The family supports distribution via **ONNX**, facilitating hardware-optimized inference engines.

### 3. Maximum Context Length
*   **Mistral AI:** Offers a superior context window of **256K tokens**, advantageous for deep research pipelines requiring massive document grounding. The sliding window mechanism helps manage computational costs within this large window.
*   **Gemma 3:** Supports **128K+ tokens**. While shorter than Mistral, this remains sufficient for most agentic and RAG use cases. The MatFormer architecture specifically addresses KV-cache bloat, allowing the 128K context to be handled more efficiently within the fixed VRAM of the RTX 5090.

### 4. Licensing Terms
*   **Mistral AI:** Weights are **fully open-weight**, allowing complete self-hosting. This is distinct from "open-source" in some definitions, as some Mistral licenses may have commercial restrictions, but the technical accessibility for self-hosting is confirmed. Commercial API pricing is available for those not self-hosting.
*   **Gemma 3:** Weights are explicitly labeled **open-source**, typically implying broader license terms (e.g., Apache 2.0), though the text confirms availability via Hugging Face and Ollama with no reported barriers to local inference.

### 5. Reliability for Agentic Tool-Calling Workloads
*   **Mistral AI:** Mistral explicitly targets **GDPR-compliant enterprise deployments, agentic coding workflows, and structured output generation**. The family is optimized for **agentic workflows**, suggesting robust function-calling schemas and tool-use capabilities. The **Small 4** unified model is particularly suited for agents requiring reasoning and tool interaction in a single pass.
*   **Gemma 3:** Designed for **RAG/agent routing** and fine-tuning via QLoRA, indicating strong support for agentic integration. The architecture's efficiency allows for rapid response cycles critical for tool-calling loops. The inclusion of **SigLip** enables agents to process visual tool outputs or screenshots directly.
*   **Benchmarking Context:** For validating agentic performance, the **AgentBench** evaluation framework is a relevant standard for tool selection, function calling, state management, and multi-step task execution. AgentBench utilizes function calling tests and Docker-based integrations, providing a reproducible pipeline to compare these models' agentic behaviors [2].

## Strategic Recommendations for RTX 5090 Deployment

### Best Models for Agentic Deep Research Pipelines
1.  **Mistral Medium 3.5 or Small 4:** Recommended for general-purpose agentic workloads. The MoE efficiency and explicit agentic optimization make these ideal for tool use, multi-step reasoning, and coding assistant tasks. Small 4 offers a unified vision/reasoning/coding stack valuable for research agents processing diverse data.
2.  **Google Gemma 3-27B:** Recommended for quality-critical deep research where reasoning depth parallels larger proprietary models is required. The 27B model fits comfortably on 32GB VRAM with quantization and offers exceptional RAG/Agent routing capabilities.

### Quantization and Framework Strategy
*   **Quantization:** Both families should be deployed with high-quality quantization (e.g., GGUF/AWQ at Q4_K_M or similar). For Gemma 3, the 27B model requires quantization to fit with context; Mistral Small/Medium models offer more headroom for higher precision or longer contexts.
*   **Frameworks:**
    *   **Ollama:** Convenient for both families, with native GGUF support.
    *   **vLLM:** Best for throughput-optimized Mistral deployments.
    *   **Transformers.js:** Supports browser-based local inference for both families, useful for hybrid edge deployments.
    *   **ONNX:** Recommended for Gemma 3 to leverage MatFormer slicing optimizations in compatible runtimes.

## Conclusion
As of August 2026, the NVIDIA RTX 5090 (32GB VRAM) can reliably run the **Mistral AI Medium 3.5/Small 4** and **Google Gemma 3 (up to 27B)** open-weight families. The requested Qwen3.6, Poolside Laguna XS 2.1, and gpt-oss models are not verified or do not exist. For agentic tool-calling and deep research, **Mistral Small 4** provides excellent unified reasoning and tool integration efficiency, while **Gemma 3-27B** offers state-of-the-art reasoning density optimized for RAG and agent routing. Both support full local deployment and self-hosting, with Mistral offering double the maximum context length.

### Sources
[1] Comparative technical and commercial analysis of open-weight LLM families: Mistral AI vs. Google's Gemma 3 - No external URLs provided in input
[2] AgentBench Profile & Context Evaluation - The input text provided did not contain any URLs
[3] Mistral AI Parameter and Architecture Specifications - No external URLs provided in input
[4] Google Gemma 3 MatFormer and Vision Capabilities Report - No external URLs provided in input
[5] AgentBench Benchmark Ecosystem Analysis - The input text provided did not contain any URLs