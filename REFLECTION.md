<!--
IMPORTANT: This is a DRAFT reflection written to match the codebase.
Read it, edit it into your own words, and adjust anything that doesn't
match your actual experience running the pipelines (real tickers tested,
real loss curves, real API errors hit, real Groq model behaviour you
observed). You will defend this in the follow-up interview, so it needs
to reflect what you actually did and observed, not just what the code
does. Keep it under 600 words after editing.
-->

# Reflection

## Architectural decisions

**Task 1** splits into three independent stages — data/indicators, LLM
reasoning, report rendering — each testable in isolation. Indicators (SMA,
RSI, MACD, Bollinger) are implemented from first principles; RSI uses
Wilder's exponential smoothing (`alpha = 1/period`), the mathematically
correct form. All Groq responses are constrained to JSON mode and
validated against Pydantic schemas with bounded retries, so a malformed
response degrades gracefully instead of crashing the pipeline.

**Task 2** uses a SEC-style risk-factor clause classifier as the domain
task — a closed 8-category taxonomy with a verifiable grounding
requirement (the explanation must cite the input clause), which makes
both dataset generation and hallucination review unambiguous. The teacher
(Groq `llama-3.3-70b-versatile`) and student (`Phi-3-mini-4k-instruct`,
fine-tuned via QLoRA) are deliberately different models. Dataset diversity
comes from sweeping sector x risk-category x writing-style combinations
rather than one repeated prompt, measured concretely via prompt-length
distribution, keyword frequency, and pairwise Jaccard near-duplicate
detection. Every fine-tuning hyperparameter is documented inline with its
reasoning rather than left at a library default. Evaluation uses an
LLM-as-judge (checking category correctness and grounding directly)
instead of BERTScore, since BERTScore would need a third model competing
for the same free T4 GPU memory without actually verifying grounding --
plus a genuinely manual hallucination-rate review, which the rubric
specifically requires not be automated.

**Task 3** makes tool selection genuinely autonomous via LangGraph's
`create_react_agent`, rather than a disguised fixed pipeline -- the
printed trace shows the model deciding at each step whether it has enough
evidence. The two-agent system enforces tool-access separation
structurally (each agent class only imports its permitted tools) and
hands off data via a typed Pydantic model rather than a raw string. The
critique loop is a simple, deterministic rule-based check rather than
another LLM call, keeping it fast and easy to verify in the trace.
Observability is one `@traced` decorator shared by every tool, so 3A, 3B,
and 3C all write to the same `agent_trace.jsonl` without duplicated
logging code.

## What I would improve with more time

For Task 1, I'd add few-shot examples to the signal-reasoning prompt, since
smaller/faster Groq models occasionally slip toward restating indicators
individually rather than truly reasoning over their combination. For Task
2, I'd generate a larger dataset (300+ examples) to better separate
adjacent categories like Market vs. Macroeconomic Risk, and swap the
rule-based RAG confidence trigger for true perplexity once I've confirmed
how to extract it cleanly from Phi-3's generation output. For Task 3, I'd
make the critique loop itself LLM-driven rather than keyword-based, and
add a second clarification round when the first still leaves a gap.

## Limitations encountered

`yfinance`'s news payload schema has shifted across versions, so
`fetch_news` handles both the old and new shapes defensively. Free-tier
Groq rate limits meant keeping temperature low and batches small during
testing. QLoRA fine-tuning on Colab's free T4 is memory-tight with a
3.8B-parameter base model; `fine_tune.py` catches CUDA OOM explicitly and
documents the batch-size/sequence-length tradeoffs to try first, per the
assessment's professional-debugging expectation. DuckDuckGo search
occasionally returns empty results for obscure tickers -- exactly the
failure mode Task 3A's "try an alternative approach" requirement targets.
