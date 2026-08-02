# CSV / Data Q&A Agent

Answers plain-English questions about a CSV file by **actually computing** over
the data with pandas — never by guessing a number from memory.

Built for the Rooman Technologies 24-Hour AI Agent Challenge (Advanced tier).

## What it does

- Loads a CSV (`data/sales_data.csv` sample included: region/quarter/product revenue data)
- Takes a question like *"Which region grew fastest last quarter?"*
- Produces the **real, computed** answer plus the pandas code/logic used to get it — so you can verify it, not just trust it

## Live test proof (LLM mode, Groq)

This agent was actually run end-to-end with a real Groq API key (`GROQ_API_KEY`,
`llama-3.3-70b-versatile` model) — not just fallback mode. All 8 sample
questions returned `"mode": "llm_generated_code"`, meaning Groq wrote the
pandas code itself and it was executed against the real CSV:

```
Q: Which region had the highest total revenue in Q4 2025?
  -> mode=llm_generated_code result=West

Q: Which region grew fastest from Q3 2025 to Q4 2025?
  -> mode=llm_generated_code result=West

Q: What is the total revenue in 2025?
  -> mode=llm_generated_code result=1557800

Q: Top 3 products by revenue
  -> mode=llm_generated_code result={'Widget A': 644000, 'Widget B': 624000, 'Widget C': 289800}

Q: What is the average revenue per region and quarter?
  -> mode=llm_generated_code result={'East-Q1': 32500.0, 'East-Q2': 34350.0, ... }

Q: Which quarter had the lowest total revenue?
  -> mode=llm_generated_code result=Q1

Q: What is the growth rate for North between Q1 2025 and Q4 2025?
  -> mode=llm_generated_code result=0.4673913043478261

Q: Which region has the most consistent revenue across quarters?
  -> mode=llm_generated_code result=East
```

Full output saved in `sample_outputs.json` (this version, generated in LLM
mode — not the earlier fallback-only version).

### How to reproduce this yourself

```bash
pip install pandas openai
export GROQ_API_KEY=your_key_here     # get a free key at console.groq.com
python agent.py --csv data/sales_data.csv --questions-file sample_questions.txt --out sample_outputs.json
```

`llm_client.py` checks for `GROQ_API_KEY` first (Groq, OpenAI-compatible
endpoint, free tier available) and falls back to `ANTHROPIC_API_KEY`
(Claude) if that's set instead. If neither is set, it drops to the
deterministic rule engine described below — so the repo is runnable either
way.

## How it avoids hallucination (the core design goal)

Two modes, chosen automatically based on whether an API key is present:

| Mode | When it runs | How it works |
|---|---|---|
| **LLM code-gen** | `ANTHROPIC_API_KEY` env var is set | Claude is shown the DataFrame's schema + a few sample rows, and asked to write a short pandas snippet that stores the answer in a variable `result`. That snippet is executed in a restricted namespace (no `open`, `import`, `exec`, network, etc.) against the real data. The number you see came from pandas running on the actual CSV, not from the model recalling/estimating it. |
| **Rule-based fallback** | No key set, or the LLM call/generated code fails | `fallback_engine.py` matches the question against a fixed set of patterns (top-N, highest/lowest by group+period, growth between two periods, variability, etc.) and answers with plain pandas `groupby`/`sum`/`std` calls. No LLM involved at all. |

Either way, the answer is a real computed value, and the response tells you which mode produced it.

## Setup

```bash
git clone <this-repo>
cd csv_qa_agent
pip install pandas          # required, always
pip install anthropic       # optional, only needed for LLM mode
export ANTHROPIC_API_KEY=sk-ant-...   # optional, only needed for LLM mode
```

Works with **zero setup beyond pandas** — without a key it just runs in
fallback mode, which is enough to reproduce every sample question below.

## Run it

Single question:
```bash
python agent.py --csv data/sales_data.csv --question "Which region had the highest total revenue in Q4 2025?"
```

Batch mode (used to generate `sample_outputs.json`):
```bash
python agent.py --csv data/sales_data.csv --questions-file sample_questions.txt --out sample_outputs.json
```

Interactive:
```bash
python agent.py --csv data/sales_data.csv
```

## Files

- `agent.py` — CLI entry point, orchestrates LLM vs. fallback mode, executes generated code safely
- `llm_client.py` — Anthropic API wrapper; builds the schema-aware prompt, strips code fences
- `fallback_engine.py` — deterministic pandas query engine (no LLM)
- `data/sales_data.csv` — sample dataset (4 regions × 4 quarters × 2–3 products, 2025)
- `sample_questions.txt` — the 8 demo questions
- `sample_outputs.json` — real output from running this repo's code against the sample data, in fallback mode (no key was used to generate this file — see Tradeoffs)

## Sample questions covered

1. Which region had the highest total revenue in Q4 2025?
2. Which region grew fastest from Q3 2025 to Q4 2025?
3. What is the total revenue in 2025?
4. Top 3 products by revenue
5. What is the average revenue per region and quarter?
6. Which quarter had the lowest total revenue?
7. What is the growth rate for North between Q1 2025 and Q4 2025?
8. Which region has the most consistent revenue across quarters?

Full computed results for all 8: see `sample_outputs.json`.

## Tradeoffs & what I'd improve with more time

- **Fallback engine is pattern-based, not general.** It only understands the
  question *shapes* it's coded for (top-N, highest/lowest-by-group,
  growth-between-two-periods, variability). Rephrase a question oddly and it
  won't match. LLM mode (confirmed working live with Groq, see proof above)
  is the general solution and handled all 8 sample questions correctly with
  freely generated code, but I haven't stress-tested it against deliberately
  weird phrasing or larger/messier CSVs — that'd be the next thing to try.
- **Code execution sandboxing is basic, not airtight.** I restrict `__builtins__`
  to a safe subset and only expose `pd`/`df`, which stops accidental file
  access or imports, but this is not a hardened sandbox (no subprocess
  isolation, no resource limits). Fine for a take-home demo against a small
  trusted CSV; not what I'd ship for untrusted user-supplied code in
  production — I'd move code execution into a separate subprocess or
  container with a timeout.
- **One retry, no self-correction loop.** If the LLM-generated snippet
  throws, I fall straight back to the rule engine rather than feeding the
  error back to the model for a second attempt. A production version should
  retry with the traceback in context before giving up.
- **Single CSV, in-memory.** No support for joins across multiple files or
  data too large to fit in memory. For bigger datasets I'd swap pandas for
  DuckDB/Polars, which the same "LLM writes the query, engine executes it"
  pattern extends to cleanly.
- **Dataset is small and synthetic (32 rows)**, built to have unambiguous
  correct answers for the 8 demo questions so the numbers can be checked by
  hand. A real submission-quality dataset would be bigger and messier.
