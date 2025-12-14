import argparse
import textwrap
from typing import Optional

import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

SYSTEM_PROMPT = """You are a senior software engineer performing a strict code review.
Your tasks:
1. Identify bugs, security vulnerabilities, and logical issues.
2. Comment on code readability, structure, and maintainability.
3. Suggest improvements. You can generate short code snippets as suggestions.
4. Be specific and strict. Refer to line numbers where necessary.

"""


def call_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


def build_prompt(code: str, filepath: Optional[str] = None) -> str:
    header = f"File: {filepath}\n" if filepath else ""
    code_block = f"```text\n{code}\n```"
    return textwrap.dedent(f"""
    {SYSTEM_PROMPT}

    {header}
    Here is the code to review:
    {code_block}

    Now provide the review.
    """)


def read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="Local AI code reviewer using Ollama.")
    parser.add_argument("file", help="Path to the source code file to review.")
    args = parser.parse_args()

    code = read_file(args.file)
    prompt = build_prompt(code, filepath=args.file)
    print("Sending code to Ollama for review...\n")
    review = call_ollama(prompt, model=MODEL_NAME)
    print("===== AI CODE REVIEW =====\n")
    print(review)
    print("\n==========================")


if __name__ == "__main__":
    main()
