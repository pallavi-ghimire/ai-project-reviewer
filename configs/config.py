OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

# CODE_EXTENSIONS = {
#     ".py", ".js", ".ts", ".tsx", ".java", ".go", ".rb", ".php", ".cs", ".cpp", ".c",
#     ".rs", ".kt", ".swift", ".sql", ".html", ".css", ".md"
# }
# Temporarily removed other extensions due to GitHub API limit
CODE_EXTENSIONS = {
    ".html", ".py", ".md"
}

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build", ".idea", ".pytest_cache"}
