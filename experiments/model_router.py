"""Route only the answer model to DeepSeek; all fast tasks and embeddings stay on Qwen."""
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agent.model_router import AnswerModelRouter


if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(Path(__file__).resolve().parents[1] / ".env.deepseek")
    try:
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com",
                        timeout=15, max_retries=0)
        print([model.id for model in client.models.list().data])
    except Exception as exc:
        print(type(exc).__name__, getattr(exc, "status_code", None))
