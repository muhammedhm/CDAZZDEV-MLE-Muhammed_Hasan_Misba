# Task 2 - Generative AI: Domain-Specific Fine-Tuning Pipeline

Open in Colab: *(after you push to GitHub, replace this with your real
badge link)*
`https://colab.research.google.com/github/<you>/CDAZZDEV-MLE-MuhammedHasanMisba/blob/main/task2_genai/task2_notebook.ipynb`

**GPU required for 2B/2C.** In Colab: `Runtime > Change runtime type > T4 GPU`
(free tier). 2A (dataset generation) and the bonus RAG layer run fine on CPU.

## Use case

SEC-style Risk-Factor Clause Classification & Explanation — see
[`problem_statement.md`](problem_statement.md) for the full structured
definition, taxonomy, and correct/incorrect criteria.

## What this does

| Stage | File | Rubric section |
|---|---|---|
| Use case + dataset engineering | `dataset_generation.py` | Task 2A (30 pts) |
| QLoRA fine-tuning | `fine_tune.py` | Task 2B (40 pts) |
| Evaluation + baseline comparison | `evaluate.py` | Task 2C (30 pts) |
| RAG fallback layer | `rag_fallback.py` | Bonus (+5 pts) |

**Teacher model:** Groq `llama-3.3-70b-versatile` (synthetic data generation).
**Student model:** `microsoft/Phi-3-mini-4k-instruct` (fine-tuned) — deliberately
a different model from the teacher.

## Quick start

```python
import os
os.environ["GROQ_API_KEY"] = "gsk_..."   # never commit this

# Task 2A - generate + split the dataset
from dataset_generation import generate_dataset, diversity_report, split_and_save
dataset = generate_dataset(min_examples=100)
print(diversity_report(dataset))
sizes = split_and_save(dataset, out_dir="data")   # writes data/{train,val,test}.jsonl
print(sizes)

# Task 2B - fine-tune (requires a CUDA/T4 runtime)
from fine_tune import run_fine_tuning
trainer, loss_history = run_fine_tuning()

# Task 2C - evaluate
from evaluate import rouge_comparison_table, judge_all, build_manual_review_sheet, compute_hallucination_rate
# ... run both models on the test set to get base_preds / ft_preds, then:
table = rouge_comparison_table(base_preds, ft_preds, references)
judged = judge_all(clauses, gold_labels, ft_preds)
sheet = build_manual_review_sheet(clauses, ft_preds)   # fill in sheet["label"] by hand, >=10 rows
rate = compute_hallucination_rate(sheet)
```

## What to verify when you run it

- **2A**: `diversity_report()` output — check `prompt_length_words` isn't
  a single repeated value, `risk_category_distribution` covers all 8
  categories reasonably evenly, and `near_duplicate_pair_ratio` is low
  (well under 1.0 — if it's high, the teacher is repeating itself; raise
  `temperature` or add more `STYLES`/`SECTORS` variety).
- **2B**: `trainer.state.log_history` must show **validation loss
  decreasing** across the 3 epochs — this is explicitly checked. If it's
  flat or increasing, see the hyperparameter comments in `fine_tune.py`
  for what to adjust first (usually learning rate or epoch count given
  such a small dataset).
- **2B**: if you hit a CUDA OOM, `fine_tune.py` catches it and prints the
  exact adjustment order to try — this doubles as your documented
  debugging practice for the submission.
- **2C**: the ROUGE-L table should show the fine-tuned model noticeably
  higher than the base model, since the base model has no reason to
  produce exact-schema JSON without fine-tuning. Fill in
  `build_manual_review_sheet()`'s `label` column yourself by reading each
  clause/prediction pair — this is the one metric in this task that is
  **not** supposed to be automated.
- **Bonus**: `rag_fallback.py`'s `__main__` block runs a CPU-only smoke
  test of the retrieval plumbing with a stub `generate` function so you
  can confirm indexing/retrieval works before wiring in the real
  fine-tuned model's `.generate()`.

## Why LLM-as-judge instead of BERTScore

BERTScore needs a third BERT-family model loaded into GPU memory
alongside the base and fine-tuned models already competing for the free
T4's 16GB, and it measures token-embedding similarity rather than the
thing this task actually cares about: is the category correct, and is the
explanation grounded in the clause (not hallucinated)? An LLM judge
checks both directly — see `JUDGE_SYSTEM_PROMPT` in `evaluate.py`.
