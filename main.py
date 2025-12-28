import argparse
import json
import textwrap
from typing import Dict, List, Optional, TypedDict, Tuple, Any

import requests
from pathlib import Path

from dataclasses import dataclass

from langgraph.graph import StateGraph, END

from mcp_client.github_client import GitHubMcpClient
from mcp_server.github_server import github_mcp_server_params

_mcp_client = None


def get_mcp_client():
    global _mcp_client
    if _mcp_client is not None:
        return _mcp_client

    params = github_mcp_server_params()
    _mcp_client = GitHubMcpClient(params)
    return _mcp_client


##################
# SETUP VARIABLES
##################
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

SECURITY_PROMPT = """You are a strict security-focused code reviewer.
Find vulnerabilities (injection, auth, secrets, unsafe deserialization, SSRF, etc.).
Return:
- High risk issues
- Medium risk issues
- Low risk issues
- Concrete fixes (short snippets ok)
Use line numbers or line ranges when possible.
"""

PERFORMANCE_PROMPT = """You are a strict performance-focused code reviewer.
Find inefficient logic, unnecessary I/O, repeated work, heavy loops, bad complexity, memory issues.
Return:
- Major bottlenecks
- Minor bottlenecks
- Concrete fixes (short snippets ok)
Use line numbers or line ranges when possible.
"""

DOCS_PROMPT = """You are a strict readability/docs/maintainability reviewer.
Find confusing naming, unclear structure, missing docstrings, poor error handling, inconsistent style.
Return:
- Maintainability risks
- Readability issues
- Suggested refactors
Use line numbers or line ranges when possible.
"""

LEAD_PROMPT = """You are a lead software engineer.
You will receive three reviews: security, performance, docs.
Your job:
1) Combine them into one clear per-file review.
2) List the top 3 issues to fix first.
3) Add a quick "overall verdict" (OK / Needs Work / Dangerous).
Be specific and concise.
"""

AGGREGATOR_PROMPT = """You are a lead engineer summarizing a repo review.
You will receive per-file reviews.
Produce:
1) Executive summary (5-10 bullet points)
2) Top 10 issues across the repo (ranked by severity/impact)
3) Hotspot files (files with most severe issues)
4) Recommended next steps (action plan)
"""


#############################
# LANGGRAPH STATES AND NODES
#############################
class ReviewState(TypedDict):
    repo_path: str
    ref: str
    files: List[str]
    current_file: Optional[str]
    per_file_results: Dict[str, Dict[str, str]]
    final_report_md: str


################
# COLLECT FILES
################
# CODE_EXTENSIONS = {
#     ".py", ".js", ".ts", ".tsx", ".java", ".go", ".rb", ".php", ".cs", ".cpp", ".c",
#     ".rs", ".kt", ".swift", ".sql", ".html", ".css", ".md"
# }
# Temporarily removed other extensions due to GitHub API limit
CODE_EXTENSIONS = {
    ".html", ".py", ".md"
}

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build", ".idea", ".pytest_cache"}


def _extract_json_from_mcp_content(resp: Any) -> Any:
    """
    Extract JSON payload from an MCP response.
    The GitHub MCP server often returns directory listings as JSON content blocks.
    """
    content = getattr(resp, "content", None)
    if content is None and isinstance(resp, dict):
        content = resp.get("content")

    if not isinstance(content, list):
        raise RuntimeError(f"Unexpected MCP response shape (no content list): {resp}")

    for item in content:
        if getattr(item, "type", None) == "json":
            return getattr(item, "json", None)
        if isinstance(item, dict) and item.get("type") == "json":
            return item.get("json")

    for item in content:
        if getattr(item, "type", None) == "text":
            t = getattr(item, "text", None)
            if isinstance(t, str):
                try:
                    return json.loads(t)
                except Exception:
                    pass
        if isinstance(item, dict) and item.get("type") == "text":
            t = item.get("text")
            if isinstance(t, str):
                try:
                    return json.loads(t)
                except Exception:
                    pass

    raise RuntimeError(f"Could not extract JSON from MCP response: {resp}")


def list_repo_paths_recursive(owner: str, repo: str, ref: str, start_path: str = "/") -> List[str]:
    """
    Recursively walk the repository tree using get_file_contents on directories.
    Returns a flat list of file paths relative to repo root.
    """
    mcp = get_mcp_client()
    to_visit = [start_path]
    files: List[str] = []

    while to_visit:
        cur = to_visit.pop()
        resp = mcp.call_tool(
            "get_file_contents",
            {"owner": owner, "repo": repo, "path": cur, "ref": ref},
        )
        data = _extract_json_from_mcp_content(resp)

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                typ = item.get("type")
                p = item.get("path") or item.get("name")
                if not isinstance(p, str):
                    continue
                if item.get("path"):
                    rel_path = item["path"].lstrip("/")
                else:
                    base = cur.strip("/")
                    rel_path = f"{base}/{p}".strip("/") if base else p.strip("/")

                if _skip(rel_path):
                    continue

                if typ == "dir":
                    to_visit.append("/" + rel_path)
                elif typ == "file":
                    files.append(rel_path)

            continue
    return sorted(set(files))


def collect_files(repo_path: str, ref: str) -> List[str]:
    if "/" not in repo_path:
        raise FileNotFoundError(f"Repo path not found (and not owner/repo): {repo_path}")

    owner, repo_name = repo_path.split("/", 1)

    all_files = list_repo_paths_recursive(owner, repo_name, ref, start_path="/")

    found_paths = [
        p for p in all_files
        if Path(p).suffix.lower() in CODE_EXTENSIONS
    ]

    found_paths = sorted(set(found_paths))
    return [f"ghmcp://{owner}/{repo_name}@{ref}/{p}" for p in found_paths]


###########################
# READ FILES FROM THE REPO
###########################


def _get(obj: Any, key: str, default=None):
    """Get attribute or dict key."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_text_from_get_file_contents(resp: Any) -> str:
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


def read_text(path: str) -> str:
    if path.startswith("ghmcp://"):
        rest = path[len("ghmcp://"):]  # owner/repo@ref/path/to/file
        owner, rest = rest.split("/", 1)  # owner | repo@ref/path/to/file
        repo_at_ref, file_path = rest.split("/", 1)  # repo@ref | path/to/file
        repo_name, ref = repo_at_ref.split("@", 1)  # repo | ref

        mcp = get_mcp_client()
        resp = mcp.call_tool(
            "get_file_contents",
            {"owner": owner, "repo": repo_name, "path": file_path, "ref": ref},
        )
        return _extract_text_from_get_file_contents(resp)

    return Path(path).read_text(encoding="utf-8", errors="replace")


####################################################################################################
# CHUNKING FILES. EACH CHUNK HAS 180 LINES. THERE IS AN OVERLAP OF 30 LINES FOR MAINTAINING CONTEXT
####################################################################################################
@dataclass
class Chunk:
    start_line: int
    end_line: int
    text: str


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


def node_collect_files(state: ReviewState) -> ReviewState:
    files = collect_files(state["repo_path"], state["ref"]) or []
    print(f"[collect_files] collected {len(files)} files")
    if len(files) == 0:
        raise RuntimeError(
            "No files collected. MCP search_code likely failed, token lacks access, "
            "or search_code output parsing is wrong."
        )
    return {
        **state,
        "files": files,
        "current_file": None,
        "per_file_results": {},
        "final_report_md": "",
    }


#################################
# REVIEW EACH FILE IN REPOSITORY
#################################
def review_file_with_agent(filepath: str, system_prompt: str, agent_name: str) -> str:
    code = read_text(filepath)
    chunks = chunk_code_by_lines(code)

    chunk_reviews: List[str] = []
    for ch in chunks:
        prompt = build_prompt(system_prompt, filepath, ch)
        chunk_reviews.append(call_ollama(prompt))

    return merge_chunk_reviews(agent_name, filepath, chunk_reviews)


def node_review_all_files(state: ReviewState) -> ReviewState:
    """
    For each file:
      - security review
      - performance review
      - docs review
      - lead merges them
    """
    results: Dict[str, Dict[str, str]] = dict(state["per_file_results"])

    for filepath in state["files"]:
        security = review_file_with_agent(filepath, SECURITY_PROMPT, "SECURITY")
        performance = review_file_with_agent(filepath, PERFORMANCE_PROMPT, "PERFORMANCE")
        docs = review_file_with_agent(filepath, DOCS_PROMPT, "DOCS")

        lead_prompt = textwrap.dedent(f"""
        {LEAD_PROMPT}

        File: {filepath}

        SECURITY REVIEW:
        {security}

        PERFORMANCE REVIEW:
        {performance}

        DOCS REVIEW:
        {docs}

        Now produce the combined per-file review. 
        IMPORTANT: Write the entire response in English only. Do not use any other language.
        Do NOT ask for further instructions. Do NOT propose what should be done.
        Directly output the final Markdown report now.
        """)
        lead = call_ollama(lead_prompt)

        results[filepath] = {
            "security": security,
            "performance": performance,
            "docs": docs,
            "lead": lead,
        }

        # basic progress print (optional)
        print(f"Reviewed: {filepath}")

    return {**state, "per_file_results": results}


def node_aggregate_repo_report(state: ReviewState) -> ReviewState:
    per_file_leads: List[str] = []
    for filepath, d in state["per_file_results"].items():
        per_file_leads.append(f"FILE: {filepath}\n{d.get('lead', '').strip()}")

    combined = "\n\n====================\n\n".join(per_file_leads)

    prompt = textwrap.dedent(f"""
    {AGGREGATOR_PROMPT}

    Here are the per-file reviews:
    {combined}

    Now write the repo-level report in Markdown.
    """)
    final_md = call_ollama(prompt)

    return {**state, "final_report_md": final_md}


###########################
# LANGGRAPH IMPLEMENTATION
###########################
def build_graph():
    g = StateGraph(ReviewState)
    g.add_node("collect_files", node_collect_files)
    g.add_node("review_all_files", node_review_all_files)
    g.add_node("aggregate", node_aggregate_repo_report)

    g.set_entry_point("collect_files")
    g.add_edge("collect_files", "review_all_files")
    g.add_edge("review_all_files", "aggregate")
    g.add_edge("aggregate", END)

    return g.compile()


##############
# CALL OLLAMA
##############
def call_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=240,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


def parse_gh_uri(uri: str) -> Tuple[str, str, str, str]:
    # gh://owner/repo@ref/path
    if not uri.startswith("gh://"):
        raise ValueError(f"Not a gh:// uri: {uri}")
    rest = uri[len("gh://"):]
    owner, rest = rest.split("/", 1)
    repo_at_ref, *maybe_path = rest.split("/", 1)
    repo, ref = repo_at_ref.split("@", 1)
    path = maybe_path[0] if maybe_path else ""
    return owner, repo, ref, path


def _extract_paths_from_search_code(resp: Any) -> List[str]:
    """
    Extract file paths from the MCP result of search_code.
    The exact JSON shape can vary; we handle the typical content->json structure.
    """
    paths: List[str] = []

    content = getattr(resp, "content", None)
    if content is None and isinstance(resp, dict):
        content = resp.get("content")

    if not isinstance(content, list):
        return paths

    for item in content:
        if getattr(item, "type", None) == "json":
            data = getattr(item, "json", None)
        elif isinstance(item, dict) and item.get("type") == "json":
            data = item.get("json")
        else:
            continue

        if isinstance(data, dict):
            items = data.get("items") or data.get("results") or data.get("matches")
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        p = it.get("path")
                        if isinstance(p, str):
                            paths.append(p)
        elif isinstance(data, list):
            for it in data:
                if isinstance(it, dict):
                    p = it.get("path")
                    if isinstance(p, str):
                        paths.append(p)

    return paths


def _skip(path: str) -> bool:
    parts = path.split("/")
    return any(p in SKIP_DIRS for p in parts)


################
# MAIN FUNCTION
################

def main():
    parser = argparse.ArgumentParser(description="Local AI code reviewer using Ollama.")
    parser.add_argument("--path", required=True, help="Path to the repo to review.")
    parser.add_argument("--ref", default="main", help="Git ref (branch/tag/SHA) to review for GitHub repos.")
    parser.add_argument("--out", default="report.md", help="Output .md report path.")
    args = parser.parse_args()

    app = build_graph()

    init_state: ReviewState = {
        "repo_path": args.path,
        "ref": args.ref,
        "files": [],
        "current_file": None,
        "per_file_results": {},
        "final_report_md": "",
    }

    final_state = app.invoke(init_state)

    out_path = Path(args.out)
    out_path.write_text(final_state["final_report_md"], encoding="utf-8")
    print(f"\nWrote report: {out_path.resolve()}")


if __name__ == "__main__":
    main()
