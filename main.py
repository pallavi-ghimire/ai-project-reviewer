import argparse
import textwrap
from typing import Dict, List, Optional, TypedDict, Any

from pathlib import Path

from langgraph.graph import StateGraph, END

from configs.aggregator_prompt import AGGREGATOR_PROMPT
from configs.docs_prompt import DOCS_PROMPT
from configs.lead_prompt import LEAD_PROMPT
from configs.performance_prompt import PERFORMANCE_PROMPT
from configs.security_prompt import SECURITY_PROMPT
from helpers.node_collect_files import collect_files
from helpers.node_review_all_files import extract_text_from_get_file_contents, chunk_code_by_lines, build_prompt, \
    merge_chunk_reviews
from mcp_client.github_client import GitHubMcpClient
from mcp_server.github_server import github_mcp_server_params
from ollama import call_ollama

#######
# INIT
#######
_mcp_client = None


def get_mcp_client():
    global _mcp_client
    if _mcp_client is not None:
        return _mcp_client

    params = github_mcp_server_params()
    _mcp_client = GitHubMcpClient(params)
    return _mcp_client


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


#################################
# COLLECT REPO FILES
#################################
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
        return extract_text_from_get_file_contents(resp)

    return Path(path).read_text(encoding="utf-8", errors="replace")


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

        print(f"Reviewed: {filepath}")

    return {**state, "per_file_results": results}


################
# CREATE REPORT
################
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
