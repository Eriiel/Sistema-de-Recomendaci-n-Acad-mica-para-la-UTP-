import requests
import os
from dotenv import load_dotenv

load_dotenv()

r = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
    json={
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": "Di hola"}],
        "max_tokens": 10
    }
)

print(r.json()["choices"][0]["message"]["content"])