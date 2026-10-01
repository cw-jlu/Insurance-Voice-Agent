"""Check connectivity to Bailian (LLM/TTS) and Tencent Cloud; write TENCENT_APP_ID to .env."""
import os, time
from pathlib import Path
from dotenv import load_dotenv

ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV)

from openai import OpenAI

c = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=os.environ["DASHSCOPE_BASE_URL"])
try:
    ms = [m.id for m in c.models.list().data]
    print(len(ms), "models:", sorted(m for m in ms if any(k in m for k in ("qwen", "embed", "tts", "asr", "deepseek"))))
except Exception as e:
    print("models err", repr(e)[:300])
for m in ("qwen-flash", "qwen-turbo", "qwen-plus"):
    try:
        t = time.perf_counter()
        r = c.chat.completions.create(model=m, messages=[{"role": "user", "content": "回复：好"}], max_tokens=5)
        print(m, "ok", round(time.perf_counter() - t, 2), r.choices[0].message.content)
    except Exception as e:
        print(m, "err", repr(e)[:200])

from tencentcloud.common import credential
from tencentcloud.cam.v20190116 import cam_client, models

try:
    cl = cam_client.CamClient(credential.Credential(os.environ["TENCENT_SECRET_ID"], os.environ["TENCENT_SECRET_KEY"]), "")
    r = cl.GetUserAppId(models.GetUserAppIdRequest())
    print("tencent appid ok", bool(r.AppId))
    if "TENCENT_APP_ID" not in ENV.read_text():
        with ENV.open("a") as f:
            f.write(f"TENCENT_APP_ID={r.AppId}\n")
except Exception as e:
    print("cam err", repr(e)[:300])
