##############
# CALL OLLAMA
##############
import requests

from configs.config import MODEL_NAME, OLLAMA_URL


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
