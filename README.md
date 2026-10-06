# CDAZZDEV-MLE-Muhammed_Hasan_Misba

Submission for the Ceylon Dazzling Dev Holding (Pvt.) Ltd. Senior Machine
Learning Engineer Technical Assessment.

Tasks completed: **all three** — Task 1 (Financial AI), Task 2 (Generative
AI fine-tuning), Task 3 (Agentic Workflows).

LLM provider: **Groq** throughout (`openai/gpt-oss-120b` by default,
configurable via `GROQ_MODEL` / `GROQ_TEACHER_MODEL` / `GROQ_JUDGE_MODEL`).

## Repository layout

```
CDAZZDEV-MLE-MuhammedHasanMisba/
├── README.md                  <- this file
├── CITATIONS.md                <- required AI-assistance / source citations
├── REFLECTION.md                <- required reflection (<=600 words, all tasks)
├── requirements.txt
├── .env.example                 <- copy to .env, fill in your own GROQ_API_KEY
├── task1_financial/
│   ├── data_pipeline.py         <- Task 1A: OHLCV + indicators + news
│   ├── llm_reasoning.py         <- Task 1B: Groq sentiment + signal reasoning
│   ├── report_generator.py      <- Bonus: Markdown/HTML brief with chart
│   └── task1_notebook.ipynb     <- Colab notebook wrapping the above
├── task2_genai/
│   ├── problem_statement.md     <- Task 2A: structured use-case definition
│   ├── dataset_generation.py    <- Task 2A: teacher-model synthetic dataset
│   ├── fine_tune.py             <- Task 2B: QLoRA 4-bit NF4 fine-tuning
│   ├── evaluate.py              <- Task 2C: ROUGE-L, LLM-judge, hallucination rate
│   ├── rag_fallback.py          <- Bonus: ChromaDB RAG fallback layer
│   ├── data/                    <- generated at runtime (train/val/test.jsonl)
│   └── task2_notebook.ipynb     <- Colab notebook wrapping the above (needs T4 GPU)
└── task3_agentic/
    ├── tools.py                 <- 5 tools + observability (traced) decorator
    ├── single_agent.py          <- Task 3A: single autonomous ReAct agent
    ├── multi_agent.py           <- Task 3B: 2-agent handoff + critique loop
    ├── memory_observability.py  <- Task 3C: session memory + persistent cache
    ├── logs/agent_trace.jsonl   <- generated at runtime (Task 3C deliverable)
    ├── cache/                   <- generated at runtime (persistent brief cache)
    └── task3_notebook.ipynb     <- Colab notebook wrapping the above
```

## Setup (Google Colab - recommended, free T4 GPU not even required for these two tasks)

1. Get a **free** Groq API key at https://console.groq.com (takes ~1 minute,
   no card required).
2. Open the relevant notebook in Colab (`task1_notebook.ipynb` or
   `task3_notebook.ipynb`) via the badge in each task's README, or upload
   this repo and open it directly.
3. In Colab, set your key as a **Secret** (key icon in the left sidebar) named
   `GROQ_API_KEY`, or run:
   ```python
   import os
   os.environ["GROQ_API_KEY"] = "gsk_..."   # paste your key, do NOT commit this
   ```
4. Run the setup cell to `pip install -r requirements.txt`.
5. Run all cells top to bottom.
6. **Task 2 only:** set the runtime to a GPU first —
   `Runtime > Change runtime type > T4 GPU` (free tier). Task 1 and Task 3
   run fine on CPU.

## Setup (local / script mode)

```bash
git clone https://github.com/muhammedhm/CDAZZDEV-MLE-MuhammedHasanMisba.git
cd CDAZZDEV-MLE-MuhammedHasanMisba
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env and paste your real GROQ_API_KEY
export $(cat .env | xargs)  # or use python-dotenv / direnv

# Task 1
cd task1_financial
python data_pipeline.py          # sanity check indicators + news for AAPL
python llm_reasoning.py          # sentiment + signal via Groq
python report_generator.py       # writes equity_brief.html

# Task 2 (2A runs anywhere; 2B/2C need a CUDA GPU)
cd ../task2_genai
python dataset_generation.py     # Task 2A - teacher-generated dataset + diversity report
python fine_tune.py              # Task 2B - QLoRA fine-tuning (GPU required)
python evaluate.py               # Task 2C - see task2_notebook.ipynb for the full interactive version
python rag_fallback.py           # Bonus - RAG fallback smoke test

# Task 3
cd ../task3_agentic
python single_agent.py           # Task 3A - autonomous single agent
python multi_agent.py            # Task 3B - two-agent handoff + critique loop
python memory_observability.py   # Task 3C - memory, cache, trace log demo
```

Swap `AAPL` for any ticker by editing the `if __name__ == "__main__":` block
at the bottom of each script, or by importing the module functions directly.

## Design notes (see REFLECTION.md for the full writeup)

- **Task 1A** computes all five indicators (SMA-50/200, RSI-14, MACD
  12/26/9, Bollinger 20/2) from first principles with pandas/numpy only —
  no TA-Lib. RSI uses Wilder smoothing via an EWM with `alpha = 1/period`,
  which is the standard, numerically-stable equivalent of Wilder's
  recursive formula.
- **Task 1B** keeps prompts as module-level string constants (separate from
  business logic), validates every LLM response against a Pydantic schema,
  and retries transient/malformed-JSON failures up to 3 times before
  logging and gracefully degrading (returns `None`, never crashes the
  pipeline).
- **Task 3A** uses LangGraph's prebuilt ReAct agent loop so tool selection
  and ordering is genuinely decided by the LLM at each step, not
  hardcoded. All 5 tools log to `agent_trace.jsonl` via a shared decorator.
- **Task 3B** enforces tool-access separation at the *class* level (Agent A
  and Agent B literally cannot call each other's tools — see `tools.py`
  imports used per class in `multi_agent.py`), hands off data via a typed
  `DataBrief` Pydantic model, and implements one clarification
  request/response/incorporate cycle.
- **Task 3C** layers short-term (in-process dict) memory over a persistent,
  date+ticker-keyed JSON cache on disk, so a second run on the same day
  skips re-executing the whole agent pipeline.
- **Task 2A** uses a *different* Groq model (`openai/gpt-oss-120b`) as
  teacher than the student being fine-tuned (`Phi-3-mini-4k-instruct`),
  per the assessment's explicit rule, and sweeps sector × risk-category ×
  writing-style combinations so the dataset doesn't collapse into
  near-duplicates of one scenario.
- **Task 2B** documents every QLoRA hyperparameter inline with the
  reasoning behind it (see the comments in `fine_tune.py`), and catches
  CUDA OOM errors with a specific, ordered remediation plan rather than
  just failing.
- **Task 2C** uses an LLM-as-judge (Groq) instead of BERTScore as the
  additional metric — see `task2_genai/README.md` for why — plus a
  genuinely manual (not automated) hallucination-rate review.
