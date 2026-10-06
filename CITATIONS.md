# Citations

Per Section 2.2 of the assessment, every instance of AI assistance is
documented here (mirrored as inline `# AI-ASSISTED:` comments at the top
of each source file).

## AI-assisted code generation

All source files in `task1_financial/` and `task3_agentic/` were scaffolded
with assistance from **Claude (claude-sonnet-4-5)**, via the Anthropic
Claude.ai chat interface, on **2026-09-10**. Representative prompts:

- "Scaffold a robust yfinance data pipeline with technical indicators
  computed from first principles and graceful handling of missing data."
  → `task1_financial/data_pipeline.py`
- "Wrap Groq chat completions with Pydantic-validated structured outputs,
  retry logic, and clean prompt/business-logic separation."
  → `task1_financial/llm_reasoning.py`
- "Render a styled HTML equity research brief with an embedded base64
  matplotlib chart." → `task1_financial/report_generator.py`
- "Implement 5 LangChain-compatible financial research tools with graceful
  error handling and a jsonl tool-call logging decorator."
  → `task3_agentic/tools.py`
- "Build a LangGraph ReAct agent over Groq that autonomously selects from
  5 financial research tools and produces a structured 3-section report."
  → `task3_agentic/single_agent.py`
- "Implement a two-agent pipeline with restricted tool access per agent, a
  Pydantic handoff schema, and one critique/clarification cycle."
  → `task3_agentic/multi_agent.py`
- "Add in-session memory, a date/ticker-keyed persistent disk cache, and
  jsonl tool-call log reporting on top of the multi-agent pipeline."
  → `task3_agentic/memory_observability.py`
- "Generate a diverse synthetic dataset for a financial risk-clause
  classification task using a Groq teacher model, with diversity metrics
  and an 80/10/10 JSONL split." → `task2_genai/dataset_generation.py`
- "Write a QLoRA 4-bit NF4 fine-tuning script for Phi-3-mini on Colab free
  tier with fully justified hyperparameters, per-epoch loss logging, OOM
  handling, and merge_and_unload for saving." → `task2_genai/fine_tune.py`
- "Build an evaluation script comparing base vs fine-tuned model with
  ROUGE-L, an LLM-as-judge structured scoring pipeline, and a manual
  hallucination-rate review workflow." → `task2_genai/evaluate.py`
- "Implement a RAG fallback that retrieves similar labeled examples from
  ChromaDB when a fine-tuned classifier self-rates low confidence, and
  re-queries with that context." → `task2_genai/rag_fallback.py`

I reviewed, tested (synthetic-data indicator checks, decorator/cache unit
tests, and live runs against my own Groq key), and modified all
AI-generated code before submission. I can explain and defend every design
decision in the follow-up interview, per Section 2's stated expectation.

## Teacher-model usage

Task 2's training dataset was synthetically generated using **Groq
`llama-3.3-70b-versatile`** as the teacher model (distinct from the
`microsoft/Phi-3-mini-4k-instruct` student model that is actually
fine-tuned, per the assessment's rule against using the same model for
both roles). The full parameterised system prompt used for generation is
defined as `TEACHER_GENERATION_PROMPT` in
`task2_genai/dataset_generation.py` and is also reproduced in
`task2_notebook.ipynb`, as required.

## Adapted open-source code / libraries

No third-party source files were copied or adapted line-for-line. The
submission depends on the following open-source libraries, used via their
public APIs as intended (standard `pip install` usage, not adaptation of
their internals):

- `yfinance` — OHLCV and news data (Task 1A)
- `groq` (official Python SDK) — LLM inference (Task 1B, Task 3 tools)
- `pydantic` — structured output validation (Task 1B, Task 3B)
- `langchain-core`, `langchain-groq`, `langgraph` — agent framework
  (`create_react_agent` prebuilt) (Task 3A)
- `duckduckgo-search` — web search tool (Task 3A/3B)
- `matplotlib`, `markdown` — report rendering (Task 1 bonus)
- `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `datasets` —
  QLoRA fine-tuning stack (Task 2B), used via their public APIs
  (`BitsAndBytesConfig`, `LoraConfig`, `SFTTrainer`, etc.), not adapted
  from any specific example script
- `rouge-score` — ROUGE-L computation (Task 2C)
- `chromadb` — local vector store for the RAG fallback bonus (Task 2)

RSI is implemented using Wilder's original smoothing method (an
exponentially-weighted moving average with `alpha = 1/period`), a
well-known, public-domain technical-analysis formula, not sourced from any
specific codebase.
