import os
import re

SYSTEM_PROMPT = """You are a data analyst assistant. You are given:
1. A pandas DataFrame called `df` (schema + a few sample rows are shown to you).
2. A plain-English question about that data.

Your job: write a SHORT pandas snippet (no imports, no function defs, no I/O,
no comments) that computes the answer and stores it in a variable named
`result`. `result` can be a number, a string, or a small DataFrame/Series.

Rules:
- Only use the variable `df` (already loaded) and the `pd` module (already imported).
- Do not read/write files, do not use input(), do not use eval/exec, do not import anything.
- Keep it to a few lines.
- Respond with ONLY the code, no explanation, no markdown fences.
"""


def _build_user_prompt(question: str, schema_description: str) -> str:
    return (
        f"DataFrame schema and sample:\n{schema_description}\n\n"
        f"Question: {question}\n\n"
        f"Write the pandas snippet now."
    )


def generate_pandas_code(question: str, schema_description: str, model: str = "claude-sonnet-4-6"):
    """
    Returns (code_str, error). error is None on success.
    Tries Groq first (GROQ_API_KEY), then Anthropic (ANTHROPIC_API_KEY).
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        return _generate_with_groq(question, schema_description, groq_key)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        return _generate_with_anthropic(question, schema_description, anthropic_key, model)

    return None, "No API key set (GROQ_API_KEY or ANTHROPIC_API_KEY)"


def _generate_with_groq(question: str, schema_description: str, api_key: str, model: str = "openai/gpt-oss-120b"):
    try:
        from openai import OpenAI  # groq uses the OpenAI-compatible SDK
    except ImportError:
        return None, "openai package not installed (pip install openai)"

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(question, schema_description)},
            ],
        )
        text = resp.choices[0].message.content
        code = _strip_code_fences(text).strip()
        return code, None
    except Exception as e:
        return None, str(e)


def _generate_with_anthropic(question: str, schema_description: str, api_key: str, model: str):
    try:
        import anthropic
    except ImportError:
        return None, "anthropic package not installed (pip install anthropic)"

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            reasoning_effort="low",
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(question, schema_description)}],
        )
        text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
        code = _strip_code_fences(text).strip()
        return code, None
    except Exception as e:
        return None, str(e)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text
