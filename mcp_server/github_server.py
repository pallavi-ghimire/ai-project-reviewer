import os
from mcp.client.stdio import StdioServerParameters
from dotenv import load_dotenv


def github_mcp_server_params() -> StdioServerParameters:
    #######################################################################
    # SET TOKEN USING $env:GITHUB_PERSONAL_ACCESS_TOKEN="ghp_access_token"
    # OR SET IT IN AN ENV FILE
    #######################################################################
    load_dotenv()
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Missing GITHUB_PERSONAL_ACCESS_TOKEN in environment.")

    return StdioServerParameters(
        command="docker",
        args=[
            "run", "-i", "--rm",
            "-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={token}",
            "-e", "GITHUB_READ_ONLY=1",
            "-e", "GITHUB_TOOLSETS=repos",
            "ghcr.io/github/github-mcp-server",
        ],
    )
