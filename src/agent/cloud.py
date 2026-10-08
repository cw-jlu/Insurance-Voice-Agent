"""Adapters for the cloud services (Alibaba Bailian LLM/TTS/embedding, Tencent Cloud real-time ASR,
OpenCode Zen Jev classifier).

Credentials come from the git-ignored ``.env`` at the repository root.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import random
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.deepseek")

FAST_MODEL = os.getenv("FAST_MODEL", "qwen-flash")
ANSWER_PROVIDER = os.getenv("ANSWER_PROVIDER", "deepseek").lower()
if ANSWER_PROVIDER not in ('qwen', 'deepseek'):
    raise ValueError('ANSWER_PROVIDER must be qwen or deepseek')
QWEN_ANSWER_MODEL = os.getenv('QWEN_ANSWER_MODEL', 'qwen-plus')
_legacy_slow = os.getenv('SLOW_MODEL', '')
SLOW_MODEL = os.getenv('ANSWER_MODEL') or (
    (_legacy_slow if _legacy_slow.startswith('deepseek-') else 'deepseek-flash')
    if ANSWER_PROVIDER == 'deepseek' else (_legacy_slow or QWEN_ANSWER_MODEL))
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3.7-text-embedding")
TTS_MODEL = os.getenv("TTS_MODEL", "qwen3-tts-flash-realtime")
TTS_VOICE = os.getenv("TTS_VOICE", "Cherry")


# --- LLM -------------------------------------------------------------------------------------------

def qwen_client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=os.environ["DASHSCOPE_BASE_URL"], timeout=30)


def llm_client():
    client = qwen_client()
    if ANSWER_PROVIDER == 'deepseek':
        from agent.model_router import AnswerModelRouter
        return AnswerModelRouter(client, SLOW_MODEL)
    return client


@dataclass
class StreamResult:
    text: str
    ttft_s: float  # time to first content token
    total_s: float
    deltas: list[tuple[float, str]] = field(default_factory=list)  # (t since start, delta)


def chat_stream(messages: list[dict], model: str, client=None, **kw) -> StreamResult:
    client = client or llm_client()
    t0 = time.perf_counter()
    ttft = None
    parts, deltas = [], []
    extra = kw.pop("extra_body", {}) or {}
    extra.setdefault("enable_thinking", False)
    stream = client.chat.completions.create(model=model, messages=messages, stream=True, extra_body=extra, **kw)
    for chunk in stream:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta.content or ""
        if d:
            now = time.perf_counter() - t0
            if ttft is None:
                ttft = now
            parts.append(d)
            deltas.append((now, d))
    total = time.perf_counter() - t0
    return StreamResult(text="".join(parts), ttft_s=ttft if ttft is not None else total, total_s=total, deltas=deltas)


def chat(messages: list[dict], model: str, client=None, **kw) -> str:
    client = client or llm_client()
    extra = kw.pop("extra_body", {}) or {}
    extra.setdefault("enable_thinking", False)
    r = client.chat.completions.create(model=model, messages=messages, extra_body=extra, **kw)
    return r.choices[0].message.content or ""


def embed(texts: list[str], client=None, batch: int = 10) -> list[list[float]]:
    client = client or llm_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        r = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + batch])
        out += [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
    return out


# --- Jev (OpenCode Zen "System One" typed classifier) ----------------------------------------------
# Jev does not generate text: it answers typed questions about a ``state`` and returns values with
# probabilities. Question types: "noul" (yes/no → probability of yes), "choice" (criteria dict →
# chosen key + probabilities), "score" (ordered criteria list → expected index + probabilities).

JEV_URL = os.getenv("JEV_URL", "https://opencode.ai/zen/v1/systemone")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-1.13-free")
JEV_TIMEOUT = float(os.getenv("JEV_TIMEOUT", "2.0"))  # P90 ≈ 0.86 s; past this the LLM fallback is faster
_jev_http = None
_jev_lock = threading.Lock()


def _jev_client():
    global _jev_http
    with _jev_lock:
        if _jev_http is None:
            import httpx

            _jev_http = httpx.Client(timeout=JEV_TIMEOUT, headers={"Authorization": f"Bearer {os.environ['OPENCODE_API_KEY']}"},
                                     limits=httpx.Limits(max_keepalive_connections=16))
        return _jev_http


def jev(state: str, questions: dict, model: str | None = None) -> dict:
    """All questions are evaluated in parallel server-side; returns ``answers`` keyed by question id."""
    r = _jev_client().post(JEV_URL, json={"model": model or JEV_MODEL, "state": state, "questions": questions})
    d = r.json()
    if r.status_code != 200 or "answers" not in d:
        raise RuntimeError(f"jev {r.status_code}: {d.get('error', d)}")
    return d["answers"]


# --- TTS (persistent realtime connection)--------------------------------------------------------

class RealtimeTTS:
    """One websocket session; each ``synth`` call sends one sentence and waits for its audio.

    Keeping the session open is the point: connection setup is paid once, not per sentence.
    """

    def __init__(self, voice: str = TTS_VOICE, model: str = TTS_MODEL, speech_rate: float | None = None):
        import dashscope
        from dashscope.audio.qwen_tts_realtime import AudioFormat, QwenTtsRealtime, QwenTtsRealtimeCallback

        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
        ws_base = os.environ["DASHSCOPE_WS_URL"].rsplit("/", 1)[0] + "/realtime"
        outer = self

        class _CB(QwenTtsRealtimeCallback):
            def on_event(self, message):
                ev = json.loads(message) if isinstance(message, str) else message
                outer._on_event(ev)

        self._events_lock = threading.Lock()
        self._audio = bytearray()
        self._first_audio_t: float | None = None
        self._done = threading.Event()
        self._error: dict | None = None
        self.client = QwenTtsRealtime(model=model, callback=_CB(), url=ws_base)
        t0 = time.perf_counter()
        self.client.connect()
        kw = {"speech_rate": speech_rate} if speech_rate else {}
        self.client.update_session(voice=voice, response_format=AudioFormat.PCM_24000HZ_MONO_16BIT, mode="commit", **kw)
        self.connect_s = time.perf_counter() - t0

    def _on_event(self, ev: dict) -> None:
        typ = ev.get("type")
        if typ == "response.audio.delta":
            with self._events_lock:
                if self._first_audio_t is None:
                    self._first_audio_t = time.perf_counter()
                self._audio += base64.b64decode(ev["delta"])
        elif typ == "response.done":
            self._done.set()
        elif typ == "error":
            self._error = ev
            self._done.set()

    def synth(self, text: str, timeout: float = 30) -> tuple[float, float, bytes]:
        """Returns (first audio s, total s, pcm 24 kHz 16-bit mono)."""
        with self._events_lock:
            self._audio = bytearray()
            self._first_audio_t = None
        self._done.clear()
        t0 = time.perf_counter()
        self.client.append_text(text)
        self.client.commit()
        if not self._done.wait(timeout):
            raise TimeoutError("tts timeout")
        if self._error:
            raise RuntimeError(self._error)
        first = (self._first_audio_t - t0) if self._first_audio_t else float("nan")
        return first, time.perf_counter() - t0, bytes(self._audio)

    def close(self) -> None:
        try:
            self.client.finish()
        except Exception:
            pass
        try:
            self.client.close()
        except Exception:
            pass


class TTSPool:
    """Several persistent sessions, so the sentences of one reply are synthesized side by side instead of one
    after another (第六轮重测: serial synthesis took 3–7 s per reply), plus audio for fixed phrases (flow
    sentences, templates) synthesized once at start-up, which costs nothing at reply time."""

    def __init__(self, n: int = 3, **kw):
        import queue

        self._kw = kw
        self.conns = [RealtimeTTS(**kw) for _ in range(n)]
        self._free: "queue.Queue[RealtimeTTS]" = queue.Queue()
        for c in self.conns:
            self._free.put(c)
        self.cache: dict[str, bytes] = {}

    def presynth(self, phrases: list[str]) -> None:
        for p in dict.fromkeys(phrases):
            if p and p not in self.cache:
                self.cache[p] = self.conns[0].synth(p)[2]

    def _one(self, text: str, t0: float) -> tuple[float, float, bytes]:
        conn = self._free.get()
        try:
            t_start = time.perf_counter() - t0
            try:
                first, total, pcm = conn.synth(text)
            except (ConnectionError, TimeoutError, RuntimeError):
                # the server closes sessions left idle (第八轮时延重测: two turns after fully cached replies failed);
                # reconnect once and retry
                conn.close()
                self.conns[self.conns.index(conn)] = conn = RealtimeTTS(**self._kw)
                first, total, pcm = conn.synth(text)
            return t_start + first, t_start + total, pcm
        finally:
            self._free.put(conn)

    def synth_all(self, sentences: list[str]) -> list[tuple[float, float, bytes, bool]]:
        """Per sentence: (first audio s, done s, pcm, cached), times measured from this call."""
        from concurrent.futures import ThreadPoolExecutor

        t0 = time.perf_counter()
        out: list = [None] * len(sentences)
        with ThreadPoolExecutor(len(self.conns)) as ex:
            futs = {}
            for i, s in enumerate(sentences):
                if s in self.cache:
                    out[i] = (0.0, 0.0, self.cache[s], True)
                else:
                    futs[i] = ex.submit(self._one, s, t0)  # submitted in order: earlier sentences start first
            for i, f in futs.items():
                out[i] = (*f.result(), False)
        return out

    def close(self) -> None:
        for c in self.conns:
            c.close()


def resample_24k_to_16k(pcm24: bytes) -> bytes:
    import numpy as np
    from scipy.signal import resample_poly

    x = np.frombuffer(pcm24, dtype=np.int16).astype(np.float32)
    y = resample_poly(x, 2, 3)
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


# --- Tencent Cloud real-time ASR -----------------------------------------------------------------

def tencent_asr_url(vad_silence_ms: int = 500, hotword_list: str | None = None) -> str:
    appid = os.environ["TENCENT_APP_ID"]
    params = {
        "secretid": os.environ["TENCENT_SECRET_ID"],
        "timestamp": int(time.time()),
        "expired": int(time.time()) + 3600,
        "nonce": random.randint(1, 10**9),
        "engine_model_type": "16k_zh",
        "voice_id": str(uuid.uuid4()),
        "voice_format": 1,  # pcm
        "needvad": 1,
        "vad_silence_time": vad_silence_ms,
        "filter_punc": 0,
        "convert_num_mode": 1,
        "word_info": 0,
    }
    if hotword_list:
        params["hotword_list"] = hotword_list
    query = "&".join(f"{k}={params[k]}" for k in sorted(params))
    raw = f"asr.cloud.tencent.com/asr/v2/{appid}?{query}"
    sig = base64.b64encode(hmac.new(os.environ["TENCENT_SECRET_KEY"].encode(), raw.encode(), hashlib.sha1).digest()).decode()
    return f"wss://{raw}&signature={urllib.parse.quote(sig)}"


@dataclass
class AsrEvent:
    t: float  # seconds since audio start
    slice_type: int  # 0 sentence start, 1 partial, 2 sentence final
    text: str


async def tencent_asr_stream(pcm16k: bytes, vad_silence_ms: int = 500, chunk_ms: int = 40,
                             trailing_silence_ms: int = 1500, on_event=None, on_audio=None) -> tuple[list[AsrEvent], float]:
    """Stream PCM at real-time pace; returns (events, audio_duration_s). Trailing silence is appended
    so the server's VAD can close the sentence, as it would in a live call."""
    import asyncio

    import websockets

    silence = b"\x00\x00" * (16000 * trailing_silence_ms // 1000)
    audio = pcm16k + silence
    speech_s = len(pcm16k) / 2 / 16000
    step = 16000 * 2 * chunk_ms // 1000
    events: list[AsrEvent] = []
    async with websockets.connect(tencent_asr_url(vad_silence_ms), max_size=None) as ws:
        hello = json.loads(await ws.recv())
        if hello.get("code") != 0:
            raise RuntimeError(hello)
        t0 = time.perf_counter()

        async def sender():
            for i in range(0, len(audio), step):
                frame = audio[i : i + step]
                if on_audio is not None:
                    on_audio(frame, (i + len(frame)) / 2 / 16000)
                await ws.send(frame)
                target = t0 + (i + step) / 2 / 16000
                await asyncio.sleep(max(0.0, target - time.perf_counter()))
            await ws.send(json.dumps({"type": "end"}))

        async def receiver():
            async for msg in ws:
                d = json.loads(msg)
                if d.get("code") != 0:
                    raise RuntimeError(d)
                if d.get("final") == 1:
                    return
                res = d.get("result") or {}
                if res:
                    event = AsrEvent(time.perf_counter() - t0, res.get("slice_type", -1), res.get("voice_text_str", ""))
                    events.append(event)
                    if on_event is not None:
                        on_event(event)  # must be nonblocking: preparation runs in a worker

        await asyncio.gather(sender(), receiver())
    return events, speech_s
