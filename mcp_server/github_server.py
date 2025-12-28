import os
from mcp.client.stdio import StdioServerParameters


def github_mcp_server_params() -> StdioServerParameters:
    #######################################################################
    # SET TOKEN USING $env:GITHUB_PERSONAL_ACCESS_TOKEN="ghp_access_token"
    #######################################################################
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Missing GITHUB_PERSONAL_ACCESS_TOKEN in environment.")

    return StdioServerParameters(
        command="docker",
        args=[
            "run", "-i", "--rm",
            "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server",
        ],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": token},
    )
