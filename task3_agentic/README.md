# Task 3 - Agentic Workflows: Multi-Agent Financial Research System

Open in Colab: *(after you push to GitHub, replace this with your real
badge link)*
`https://colab.research.google.com/github/<you>/CDAZZDEV-MLE-MuhammedHasanMisba/blob/main/task3_agentic/task3_notebook.ipynb`

## What this does

| Stage | File | Rubric section |
|---|---|---|
| 5 tools + observability decorator | `tools.py` | shared by 3A/3B/3C |
| Single autonomous agent | `single_agent.py` | Task 3A (50 pts) |
| Two-agent handoff + critique loop | `multi_agent.py` | Task 3B (35 pts) |
| Memory + persistent cache | `memory_observability.py` | Task 3C (15 pts) |

## Quick start

```python
import os
os.environ["GROQ_API_KEY"] = "gsk_..."   # never commit this

# Task 3A
from single_agent import run_single_agent
out_a = run_single_agent("AAPL")
print(out_a["final_report"])

# Task 3B
from multi_agent import run_multi_agent
out_b = run_multi_agent("AAPL")
print(out_b["final_report"])

# Task 3C
from memory_observability import ResearchSession, print_trace_log_summary
session = ResearchSession()
session.research("AAPL")                     # live run
session.research("AAPL")                     # short-term memory hit
print(session.ask_followup("AAPL", "How volatile has it been?"))
print_trace_log_summary()
```

## What to verify when you run it

- **3A - autonomous tool selection**: read the printed trace. Confirm the
  order of tool calls differs sensibly between two different tickers (e.g.
  a ticker with thin news coverage should trigger more `web_search` calls)
  — that's your evidence this isn't hardcoded.
- **3A - error handling**: temporarily break `get_news` (e.g. return an
  error dict unconditionally) and confirm the agent still produces a
  report using the remaining tools instead of crashing.
- **3B - restricted tool access**: `DataAnalystAgent` only ever calls
  `get_price_data` / `calculate_volatility`; `ResearchWriterAgent` only
  ever calls `get_news` / `web_search`. This is enforced structurally
  (each class only imports/uses its allowed tools) — grep the file to
  confirm no cross-calls exist.
- **3B - critique loop**: check `out_b["trace"]` for a
  `request_clarification` → `answer_clarification` pair.
- **3C**: after your first run, check that `task3_agentic/cache/` contains
  a `TICKER_YYYY-MM-DD.json` file, and that `task3_agentic/logs/
  agent_trace.jsonl` grows with one line per tool call (name, inputs,
  truncated output, duration).

## Note on tool call counts

Every tool call across `single_agent.py`, `multi_agent.py`, and
`memory_observability.py` writes to the **same**
`task3_agentic/logs/agent_trace.jsonl`, since they share `tools.py`. Delete
that file before a clean demo run if you want an isolated trace per task.
