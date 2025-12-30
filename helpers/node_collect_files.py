import json
from pathlib import Path
from typing import List, Any

from configs.config import CODE_EXTENSIONS, SKIP_DIRS
from mcp_client.github_client import GitHubMcpClient
from mcp_server.github_server import github_mcp_server_params


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


def _skip(path: str) -> bool:
    parts = path.split("/")
    return any(p in SKIP_DIRS for p in parts)


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
