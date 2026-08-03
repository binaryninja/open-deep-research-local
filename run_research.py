"""Headless runner for the deep researcher graph with a local Ollama model."""
import asyncio
import logging
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from langchain_core.messages import HumanMessage

from open_deep_research.deep_researcher import deep_researcher

QUESTION = (
    "What are the current state-of-the-art open-weight LLMs that can run locally on a "
    "single NVIDIA RTX 5090 (32GB VRAM) as of August 2026? Compare the leading options "
    "(e.g. Qwen3.6 family, Poolside Laguna XS 2.1, gpt-oss, and any other notable "
    "releases) on quality, speed, context length, licensing, and suitability for "
    "agentic tool-calling workloads such as deep research pipelines."
)


async def main():
    question = sys.argv[1] if len(sys.argv) > 1 else QUESTION
    logging.info("Starting deep research: %s", question[:120])

    final_state = None
    async for event in deep_researcher.astream(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 100},
        stream_mode="values",
    ):
        final_state = event
        keys = [k for k, v in event.items() if v]
        logging.info("state update, populated keys: %s", keys)

    report = (final_state or {}).get("final_report", "")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = f"report-{stamp}.md"
    with open(out_path, "w") as f:
        f.write(report if report else "NO REPORT GENERATED\n\nFinal state keys: " + str(list((final_state or {}).keys())))
    logging.info("Report written to %s (%d chars)", out_path, len(report))


asyncio.run(main())
