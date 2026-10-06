# Task 2 - Use Case Definition

## Use case
**SEC-style Risk-Factor Clause Classification & Explanation.**

Public companies disclose a "Risk Factors" section in their 10-K/10-Q
filings — paragraphs of free text describing threats to the business.
Analysts and compliance teams need to triage these at scale: categorize
each clause against a standard risk taxonomy, assign a severity, and get
a short, specific explanation grounded in the actual text (not a generic
restatement). A general-purpose chatbot tends to either hedge
("this could be a risk because...") or hallucinate specifics not present
in the clause. This is exactly the kind of narrow, high-precision,
domain-specific task fine-tuning is suited for — and it extends naturally
from the Task 1 financial-AI theme.

## Input
A single paragraph of free text (50-200 words), written in the style of a
company's Risk Factors disclosure, covering one specific risk.

## Output
A single JSON object:
```json
{
  "risk_category": "<one of the 8 taxonomy labels below>",
  "severity": "low" | "medium" | "high",
  "explanation": "<one sentence, must reference specific content from the clause>"
}
```

### Risk taxonomy (closed set — this is what makes grading well-defined)
`Market Risk`, `Credit Risk`, `Liquidity Risk`, `Operational Risk`,
`Regulatory/Legal Risk`, `Cybersecurity Risk`, `Reputational Risk`,
`Macroeconomic Risk`.

## What counts as correct vs. incorrect

| | Correct | Incorrect |
|---|---|---|
| **Category** | Matches the taxonomy label a human analyst would assign given the clause's dominant theme | Wrong category, or a category outside the fixed taxonomy |
| **Severity** | Reasonable given the clause's own language (e.g. "could materially harm" → high; "may modestly affect" → low) | Severity contradicts the clause's own stated magnitude |
| **Explanation** | References a specific phrase or fact *from the clause* | Generic boilerplate that could apply to any clause, or invents a fact/number not present in the input (**hallucination**) |
| **Format** | Valid JSON matching the schema exactly | Malformed JSON, extra fields, wrong types |

This gives Task 2C's hallucination-rate review an unambiguous test: does
the explanation cite something that is actually in the input clause?

## Why not a generic chatbot / creative-writing task
This task has a closed label set, a verifiable grounding requirement (the
explanation must cite the input), and a realistic enterprise use case
(compliance/analyst triage), which is what the assessment's Section on
Task 2A use-case quality is explicitly looking for.
