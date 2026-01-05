# AI-Powered Repository Code Reviewer Agent

This project is a local, agent-based AI code review system that automatically analyzes a software repository and produces a comprehensive Markdown review report. It leverages LangGraph for orchestration, Ollama for local LLM inference, and GitHub MCP for reviewing remote repositories (REST implementation in main_rest.py).

The system performs per-file security, performance, and documentation reviews, merges them into structured file-level reports, and then aggregates everything into a single repository-level report.

## Key Features

- Multi-agent review pipeline

  - Security analysis

  - Performance analysis

  - Documentation quality review

- LangGraph-based workflow

  - Deterministic, inspectable execution graph

- Chunked code analysis

  - Large files are reviewed safely and consistently

- Local LLM execution

  - Uses Ollama; no cloud APIs required

- Supports local and GitHub repositories

  - GitHub access via MCP (Model Context Protocol)

- Markdown output 

  - Ready for sharing, auditing, or CI artifacts


## High-Level Architecture

The review process follows a fixed graph:

1. Collect files

   - Enumerates repository files (local or GitHub)

2. Per-file review

   - Security agent

   - Performance agent

   - Documentation agent

   - Lead agent merges the above into a single per-file report

3. Repository aggregation

   - Combines all per-file reports into a final Markdown document


## Requirements
- Python

  - Python 3.10+

- Python Dependencies

  - langgraph

  - ollama

  - typing_extensions (if using older Python versions)
  
The other requirements can be found in requirements.txt file. 


## Ollama Setup

This project assumes:

- Ollama is installed and running locally

- At least one LLM model is available (e.g., llama3, mistral, etc.). Update the variables in the configs > config.py file accordingly.

- GITHUB_PERSONAL_ACCESS_TOKEN is stored in an environment (.env) file 



## Review Project

### Review a local repo
Run:
```
python main_rest.py \
  --path /path/to/local/repo \
  --out report.md 
```

### Review a repo via GitHub REST API
Run:
```
python main_rest.py \
  --path username/project_name \
  --out report.md 
```

### Review a repo via GitHub MCP
Run:
```
python main.py \
  --path username/project_name \
  --out report.md 
```
Note: --out is optional; by default the report is generated in report.md.
