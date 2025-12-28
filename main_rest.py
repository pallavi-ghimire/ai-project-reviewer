import argparse
import base64
import os
import textwrap
from typing import Dict, List, Optional, TypedDict

import requests
from pathlib import Path

from dataclasses import dataclass

from langgraph.graph import StateGraph, END


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
    ".html"
}

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build", ".idea", ".pytest_cache"}


def collect_files(repo_path: str) -> List[str]:
    repo = Path(repo_path)

    ###################
    # LOCAL FILESYSTEM
    ###################
    if repo.exists():
        if not repo.is_dir():
            raise ValueError(f"Repo path is not a directory: {repo_path}")

        found: List[str] = []
        for root, dirs, files in os.walk(repo):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in files:
                p = Path(root) / name
                if p.suffix.lower() in CODE_EXTENSIONS:
                    if p.stat().st_size > 300_000:
                        continue
                    found.append(str(p))
        found.sort()
        return found

    #######################
    # GITHUB WITH REST API
    #######################
    if "/" not in repo_path:
        raise FileNotFoundError(f"Repo path not found: {repo_path}")

    owner, repo_name = repo_path.split("/", 1)

    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"

    r = requests.get(f"https://api.github.com/repos/{owner}/{repo_name}", headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch repo metadata: {r.status_code} {r.text}")
    default_branch = r.json().get("default_branch", "master")

    r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo_name}/branches/{default_branch}",
        headers=headers,
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch branch '{default_branch}': {r.status_code} {r.text}")
    sha = r.json()["commit"]["sha"]

    r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{sha}",
        params={"recursive": "1"},
        headers=headers,
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch tree: {r.status_code} {r.text}")

    tree = r.json().get("tree", [])
    found_paths = []
    for item in tree:
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        if not path:
            continue
        if any(part in SKIP_DIRS for part in path.split("/")):
            continue
        if Path(path).suffix.lower() in CODE_EXTENSIONS:
            found_paths.append(path)

    found_paths.sort()
    return [f"ghrest://{owner}/{repo_name}@{default_branch}/{p}" for p in found_paths]


###########################
# READ FILES FROM THE REPO
###########################
def read_text(path: str) -> str:
    if path.startswith("ghrest://"):
        rest = path[len("ghrest://"):]
        # owner/repo@branch/path/to/file
        owner, rest = rest.split("/", 1)
        repo_and_branch, file_path = rest.split("/", 1)
        repo_name, branch = repo_and_branch.split("@", 1)

        token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
        headers = {}
        if token:
            headers["Authorization"] = f"token {token}"

        r = requests.get(
            f"https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}",
            params={"ref": branch},
            headers=headers,
            timeout=30,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Failed to read {file_path}: {r.status_code} {r.text}")

        data = r.json()
        if data.get("encoding") == "base64" and isinstance(data.get("content"), str):
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")

        raise RuntimeError(f"Unexpected contents response for {file_path}")


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
    files = collect_files(state["repo_path"])
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
    # Prepare a compact input to the final aggregator (use lead summaries, not all raw text).
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


################
# MAIN FUNCTION
################

def main():
    parser = argparse.ArgumentParser(description="Local AI code reviewer using Ollama.")
    parser.add_argument("--path", required=True, help="Path to the repo to review.")
    parser.add_argument("--out", default="report.md", help="Output .md report path.")
    args = parser.parse_args()

    app = build_graph()

    init_state: ReviewState = {
        "repo_path": args.path,
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
