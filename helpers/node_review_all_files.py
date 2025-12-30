import textwrap
from dataclasses import dataclass
from typing import List, Any


####################################################################################################
# CHUNKING FILES. EACH CHUNK HAS 180 LINES. THERE IS AN OVERLAP OF 30 LINES FOR MAINTAINING CONTEXT
####################################################################################################
@dataclass
class Chunk:
    start_line: int
    end_line: int
    text: str


def _get(obj: Any, key: str, default=None):
    """Get attribute or dict key."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_text_from_get_file_contents(resp: Any) -> str:
    content = _get(resp, "content")
    if content is None and isinstance(resp, dict):
        content = resp.get("content")

    if not isinstance(content, list):
        raise RuntimeError(f"Unexpected get_file_contents response shape: {resp!r}")

    candidates: List[str] = []

    for item in content:
        resource = _get(item, "resource")
        resource_text = _get(resource, "text")
        if isinstance(resource_text, str) and resource_text.strip():
            candidates.append(resource_text)
            continue

        if _get(item, "type") == "json":
            j = _get(item, "json")
            j_content = _get(j, "content")
            if isinstance(j_content, str) and j_content.strip():
                candidates.append(j_content)
                continue

        if _get(item, "type") == "text":
            t = _get(item, "text")
            if isinstance(t, str) and t.strip():
                candidates.append(t)
                continue

    if not candidates:
        raise RuntimeError(f"Could not extract text from get_file_contents response: {resp!r}")

    def score(s: str) -> int:
        base = len(s)
        if "\n" in s:
            base += 500
        if "SHA:" in s or "successfully downloaded" in s.lower():
            base -= 2000
        return base

    return max(candidates, key=score)


def chunk_code_by_lines(code: str, max_lines: int = 180, overlap: int = 30) -> List[Chunk]:
    lines = code.splitlines()
    chunks: List[Chunk] = []

    i = 0
    n = len(lines)
    while i < n:
        start = i
        end = min(i + max_lines, n)
        chunk_lines = lines[start:end]
        chunk_text = "\n".join(chunk_lines)
        chunks.append(Chunk(start_line=start + 1, end_line=end, text=chunk_text))

        if end == n:
            break
        i = end - overlap
        if i < 0:
            i = 0

    return chunks


def build_prompt(system_prompt: str, filepath: str, chunk: Chunk) -> str:
    return textwrap.dedent(f"""
    {system_prompt}

    File: {filepath}
    Chunk lines: {chunk.start_line}-{chunk.end_line}

    ```text
    {chunk.text}
    ```

    Review this chunk. Be specific and strict.
    """)


def merge_chunk_reviews(agent_name: str, filepath: str, chunk_reviews: List[str]) -> str:
    """
    Merge chunk-level reviews into one file-level review for that agent.
    Concatenate with separators.
    """
    joined = "\n\n---\n\n".join(chunk_reviews).strip()
    return textwrap.dedent(f"""
    [{agent_name} REVIEW] {filepath}

    {joined}
    """).strip()

