# Task 1 - Financial AI: LLM-Powered Equity Research Assistant

Open in Colab: *(after you push to GitHub, replace this with your real
badge link, e.g.)*
`https://colab.research.google.com/github/<you>/CDAZZDEV-MLE-MuhammedHasanMisba/blob/main/task1_financial/task1_notebook.ipynb`

## What this does

| Stage | File | Rubric section |
|---|---|---|
| Data pipeline | `data_pipeline.py` | Task 1A (60 pts) |
| LLM sentiment + signal reasoning | `llm_reasoning.py` | Task 1B (40 pts) |
| Rendered research brief | `report_generator.py` | Bonus (+5 pts) |

## Quick start

```python
import os
os.environ["GROQ_API_KEY"] = "gsk_..."   # never commit this

from data_pipeline import run_pipeline
from llm_reasoning import run_reasoning
from report_generator import render_html

pipeline_out = run_pipeline("AAPL")          # Task 1A
reasoning_out = run_reasoning(pipeline_out)  # Task 1B
html = render_html(pipeline_out, reasoning_out)  # Bonus
with open("equity_brief.html", "w") as f:
    f.write(html)
```

## What to verify when you run it

- `pipeline_out["summary"]` has all 5 required fields populated (or `None`
  gracefully, never a crash) for a ticker with sparse data.
- `pipeline_out["ohlcv_with_indicators"]` has `sma_50`, `sma_200`, `rsi_14`,
  `macd`, `signal`, `bb_upper`, `bb_mid`, `bb_lower` columns.
- `reasoning_out["sentiment"]["per_headline"]` — each item has `headline`,
  `sentiment`, `confidence`, `brief_reason`.
- `reasoning_out["signal"]["justification"]` reasons over the *combination*
  of indicators (trend + momentum + volatility), not just a value dump —
  read it critically; if it just restates numbers, tighten
  `SIGNAL_SYSTEM_PROMPT` in `llm_reasoning.py`.
- Try an intentionally bad ticker (e.g. `"NOTAREALTICKER"`) to confirm
  `DataFetchError` is raised cleanly rather than an unhandled traceback.
