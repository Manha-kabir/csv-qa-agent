#!/usr/bin/env python3
"""
CSV / Data Q&A Agent
--------------------
Answers plain-English questions about a CSV file by actually computing over
the data (never by guessing a number from the model's memory).

Two execution modes, chosen automatically:

1. LLM mode (used if ANTHROPIC_API_KEY/ GROQ_API_KEY is set):
   Claude is asked to write a short pandas snippet that computes the answer
   into a variable `result`. The snippet is executed in a restricted
   namespace (only `pd` and `df` available, no builtins like open/exec/import)
   so the model can answer arbitrary questions about the data.

2. Fallback mode (used if no API key, or the LLM call fails):
   A deterministic, regex/keyword based engine (fallback_engine.py) handles
   a fixed set of common question shapes with plain pandas. No LLM involved,
   so this mode is 100% reproducible and needs zero setup.

Every answer shows the actual computed number/table, not a paraphrase.

Usage:
    python agent.py --csv data/sales_data.csv --question "Which region had the highest total revenue in Q4 2025?"
    python agent.py --csv data/sales_data.csv --questions-file sample_questions.txt --out sample_outputs.json
    python agent.py --csv data/sales_data.csv           # interactive loop
"""

import argparse
import json
import sys

import pandas as pd

import fallback_engine
import llm_client

SAFE_BUILTINS = {"len": len, "sum": sum, "min": min, "max": max, "round": round, "sorted": sorted}


def describe_schema(df: pd.DataFrame) -> str:
    lines = [f"Columns and dtypes:\n{df.dtypes.to_string()}", "\nFirst 3 rows:", df.head(3).to_string()]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        uniques = df[col].unique()
        if len(uniques) <= 15:
            lines.append(f"\nUnique values in '{col}': {list(uniques)}")
    return "\n".join(lines)


def run_generated_code(code: str, df: pd.DataFrame):
    """Execute LLM-generated pandas code in a restricted namespace."""
    namespace = {"pd": pd, "df": df, "__builtins__": SAFE_BUILTINS}
    exec(code, namespace)  # noqa: S102 - restricted namespace, sandboxed demo use only
    if "result" not in namespace:
        raise ValueError("Generated code did not set a `result` variable.")
    return namespace["result"]


def answer_question(question: str, df: pd.DataFrame, schema_desc: str, verbose: bool = True):
    code, err = llm_client.generate_pandas_code(question, schema_desc)

    if code:
        try:
            result = run_generated_code(code, df)
            return {
                "question": question,
                "mode": "llm_generated_code",
                "code": code,
                "result": _jsonable(result),
            }
        except Exception as e:
            if verbose:
                print(f"  [!] LLM-generated code failed to run ({e}); falling back to rule engine.")

    result, explanation = fallback_engine.answer(question, df)
    return {
        "question": question,
        "mode": "fallback_rule_engine",
        "explanation": explanation,
        "result": _jsonable(result),
        "llm_unavailable_reason": err if not code else "generated code raised an exception",
    }


def _jsonable(result):
    if isinstance(result, (pd.Series, pd.DataFrame)):
        return json.loads(result.to_json())
    if isinstance(result, (pd.Timestamp,)):
        return str(result)
    if isinstance(result, dict):
        return {k: _jsonable(v) for k, v in result.items()}
    if hasattr(result, "item"):  # numpy scalar types (np.int64, np.float64, ...)
        return result.item()
    return result


def main():
    parser = argparse.ArgumentParser(description="CSV Q&A Agent")
    parser.add_argument("--csv", required=True, help="Path to the CSV file to load")
    parser.add_argument("--question", help="A single question to answer")
    parser.add_argument("--questions-file", help="A text file with one question per line")
    parser.add_argument("--out", help="Where to write JSON results (used with --questions-file)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    schema_desc = describe_schema(df)

    if args.question:
        result = answer_question(args.question, df, schema_desc)
        print(json.dumps(result, indent=2))
        return

    if args.questions_file:
        with open(args.questions_file) as f:
            questions = [line.strip() for line in f if line.strip()]
        results = []
        for q in questions:
            print(f"Q: {q}")
            r = answer_question(q, df, schema_desc)
            print(f"  -> mode={r['mode']} result={r['result']}\n")
            results.append(r)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Wrote {len(results)} answers to {args.out}")
        return

    # Interactive loop
    print("CSV Q&A Agent. Type a question, or 'quit' to exit.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if q.lower() in {"quit", "exit"}:
            break
        if not q:
            continue
        r = answer_question(q, df, schema_desc)
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    sys.exit(main())
