"""
Task 3A - Single tool-using research agent (LangGraph ReAct-style loop).
The LLM decides autonomously which of the 5 tools to call and in what
order, based on what it observes -- there is no hardcoded call sequence.

# AI-ASSISTED: Claude (claude-sonnet-4-5), Prompt: 'build a LangGraph
# ReAct agent over Groq that autonomously selects from 5 financial
# research tools and produces a structured 3-section report',
# Date: 2026-09-10
"""

from __future__ import annotations

import os

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from tools import ALL_TOOLS

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are a senior equity research analyst agent. You have five tools: \
get_price_data, get_news, calculate_volatility, llm_sentiment, and web_search. \
Decide autonomously which tools to call and in what order based on what you observe -- \
do not follow a fixed sequence, and do not call a tool you don't need. If a tool returns \
an 'error' key or an empty result, try an alternative tool or a reformulated query rather \
than giving up or stopping.

When you have gathered enough evidence, produce a FINAL structured report with exactly
these three sections and nothing else:
### Financial Health Summary
### Top Three Risks
(each risk must cite the specific data point from a tool result that supports it)
### Hedge Strategy Recommendation
"""


def build_agent():
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.3)
    # langgraph renamed the system-prompt kwarg from `state_modifier` (older
    # releases) to `prompt` (newer releases, pre-1.0 API). Try the current
    # name first and fall back so this works across installed versions.
    try:
        return create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)
    except TypeError:
        return create_react_agent(llm, ALL_TOOLS, state_modifier=SYSTEM_PROMPT)


def run_single_agent(ticker: str) -> dict:
    agent = build_agent()
    query = (
        f"Analyse the current financial health and market sentiment of {ticker}. "
        "Identify the top three risks to its share price over the next 90 days and "
        "suggest one data-driven hedge strategy."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    # Full observe -> decide -> act trace (Task 3A requires at least one
    # visible cycle of: tool call, observe result, decide next action).
    trace = []
    for msg in result["messages"]:
        trace.append({
            "role": getattr(msg, "type", msg.__class__.__name__),
            "content": getattr(msg, "content", ""),
            "tool_calls": getattr(msg, "tool_calls", None),
        })

    final_report = result["messages"][-1].content
    return {"ticker": ticker.upper(), "trace": trace, "final_report": final_report}


if __name__ == "__main__":
    out = run_single_agent("AAPL")
    print("\n--- MESSAGE TRACE (observe -> decide -> act) ---")
    for step in out["trace"]:
        preview = (step["content"] or "")[:200]
        print(f"[{step['role']}] {preview}")
        if step["tool_calls"]:
            print("  tool_calls:", step["tool_calls"])
    print("\n--- FINAL REPORT ---")
    print(out["final_report"])
