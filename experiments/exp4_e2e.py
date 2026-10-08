"""实验 4：端到端语音时延（真实云服务）。

每一轮：
1. 用阿里云 TTS 合成一段"用户语音"（男声），按实时速度推给腾讯云实时 ASR；
2. ASR 给出整句结果后，立即播放本地预合成的承接语（首句快车道，规则选择，不调模型）；
3. 同时运行慢思考（理解 → 查表/检索 → 流式生成 → 逐句护栏），每句通过护栏后交给持久连接的 TTS（女声）；
4. 按真实时间把用户音频和 Agent 音频放进双声道录音（左：用户，右：Agent），并记录时间戳。

所有时延从"用户说完话"（由音频能量判断）开始计算，与官方用录音验收的口径一致。
"""

from __future__ import annotations

import asyncio
import json
import queue
import statistics
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agent import cloud  # noqa: E402
from agent.pipeline import HybridRetriever, Pipeline, stream_answer  # noqa: E402
from embed_cache import corpus, load  # noqa: E402

OUT = ROOT / "experiments" / "results"
REC = ROOT / "experiments" / "recordings"
SR_OUT = 24000  # recording sample rate

UTTERANCES = [
    ("报价", "我三十岁，有社保，一年多少钱？", {}),
    ("病史", "我有高血压好几年了，一直在吃降压药，买这个以后住院能报吗？", {}),
    ("免赔额", "我自己先掏一万，剩下的是不是全报？", {}),
    ("保障范围", "刷到广告说几百万保额，具体都能报什么？", {}),
    ("价格变化", "以后每年都是这个价吗？", {"age": 30, "has_social": True}),
    ("身份", "你是真人还是机器人？", {}),
    ("等待期", "等待期是多久？", {}),
    ("加油包", "门急诊加油包多大年纪能买？", {}),
]
OPENER = "好的，我帮您看一下。"
SPEECH_RATE = 1.20


def speech_end_s(pcm16k: bytes, frame_ms: int = 20, thresh_db: float = -40.0) -> float:
    x = np.frombuffer(pcm16k, dtype=np.int16).astype(np.float32) / 32768
    n = 16000 * frame_ms // 1000
    frames = x[: len(x) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    voiced = np.where(db > thresh_db)[0]
    return (voiced[-1] + 1) * frame_ms / 1000 if len(voiced) else 0.0


def to24k(pcm16k: bytes) -> np.ndarray:
    from scipy.signal import resample_poly

    x = np.frombuffer(pcm16k, dtype=np.int16).astype(np.float32)
    return resample_poly(x, 3, 2)


def run_turn(name: str, text: str, known: dict, pipe: Pipeline, user_tts: cloud.RealtimeTTS,
             agent_tts: cloud.RealtimeTTS, opener_pcm: bytes, rec_path: Path) -> dict:
    _, _, user24 = user_tts.synth(text)
    user16 = cloud.resample_24k_to_16k(user24)
    t_end = speech_end_s(user16)

    events, _ = asyncio.run(cloud.tencent_asr_stream(user16, vad_silence_ms=500))
    finals = [e for e in events if e.slice_type == 2]
    if not finals:
        raise RuntimeError(f"no final ASR result for {text}")
    final = finals[-1]
    t_final = final.t
    asr_text = "".join(e.text for e in finals)
    first_partial = next((e for e in events if e.text), None)

    # --- after the final ASR result: opener plays immediately; slow lane runs ---
    agent_track: list[tuple[float, bytes, str]] = []  # (start time from audio t=0, pcm24k, text)
    opener_used = name != "身份"
    if opener_used:
        agent_track.append((t_final, opener_pcm, OPENER))
    cursor = t_final + (len(opener_pcm) / 2 / SR_OUT if opener_used else 0.0)

    q: queue.Queue = queue.Queue()
    tts_log: list[dict] = []

    def tts_worker():
        while True:
            item = q.get()
            if item is None:
                return
            sent, t_ready = item
            first, total, pcm = agent_tts.synth(sent)
            tts_log.append({"text": sent, "ready": t_ready, "first_audio": t_ready + first,
                            "audio_done": t_ready + total, "pcm": pcm})

    worker = threading.Thread(target=tts_worker, daemon=True)
    worker.start()
    sentences, dropped, understood_at = [], [], None
    gen_start = time.perf_counter()
    for kind, payload, t in stream_answer(pipe, asr_text, known):
        t_abs = t_final + (time.perf_counter() - gen_start)
        if kind == "understood":
            understood_at = t_abs
        elif kind == "sentence":
            sentences.append({"text": payload, "t": t_abs})
            q.put((payload, t_abs))
        elif kind == "dropped":
            dropped.append(payload)
    gen_done = t_final + (time.perf_counter() - gen_start)
    q.put(None)
    worker.join()

    # Schedule playback: each sentence starts when its first audio arrives and the previous one ended.
    first_answer_audio = None
    for log in tts_log:
        start = max(cursor, log["first_audio"])
        first_answer_audio = first_answer_audio if first_answer_audio is not None else start
        agent_track.append((start, log["pcm"], log["text"]))
        cursor = start + len(log["pcm"]) / 2 / SR_OUT

    # Stereo recording: left = user, right = agent.
    total_s = max(len(user16) / 2 / 16000, cursor) + 0.5
    left = np.zeros(int(total_s * SR_OUT), dtype=np.float32)
    right = np.zeros_like(left)
    u = to24k(user16)
    left[: len(u)] = u[: len(left)]
    for start, pcm, _ in agent_track:
        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        i = int(start * SR_OUT)
        right[i : i + len(a)] += a[: max(0, len(right) - i)]
    stereo = np.stack([left, right], axis=1).clip(-32768, 32767).astype(np.int16)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(rec_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR_OUT)
        w.writeframes(stereo.tobytes())

    first_response = (t_final if opener_used else first_answer_audio) - t_end
    return {
        "name": name,
        "user_text": text,
        "asr_text": asr_text,
        "speech_end_s": round(t_end, 3),
        "asr_endpoint_s": round(t_final - t_end, 3),
        "first_partial_s": round(first_partial.t, 3) if first_partial else None,
        "first_response_s": round(first_response, 3),
        "understood_after_end_s": round(understood_at - t_end, 3) if understood_at else None,
        "first_substantive_audio_s": round(first_answer_audio - t_end, 3) if first_answer_audio else None,
        "answer_audio_all_ready_s": round(max(l["audio_done"] for l in tts_log) - t_end, 3) if tts_log else None,
        "answer_playback_end_s": round(cursor - t_end, 3),
        "generation_done_s": round(gen_done - t_end, 3),
        "sentences": [s["text"] for s in sentences],
        "dropped": dropped,
        "recording": str(rec_path.relative_to(ROOT)),
        "timeline": [{"start_s": round(s - t_end, 3), "text": t} for s, _, t in agent_track],
    }


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))], 3)


def main(rounds: int = 2) -> dict:
    client = cloud.llm_client()
    chunks = corpus(False)
    pipe = Pipeline(HybridRetriever(chunks, load(chunks, "clauses"), client=client), client=client)
    user_tts = cloud.RealtimeTTS(voice="Ethan")
    agent_tts = cloud.RealtimeTTS(voice="Cherry", speech_rate=SPEECH_RATE)
    _, _, opener_pcm = agent_tts.synth(OPENER)  # pre-synthesized once, cached for the session
    turns = []
    try:
        for r in range(rounds):
            for name, text, known in UTTERANCES:
                for attempt in range(3):
                    try:
                        turns.append(run_turn(name, text, known, pipe, user_tts, agent_tts, opener_pcm,
                                              REC / f"v3_r{r + 1}_{name}.wav"))
                        break
                    except Exception as e:
                        if attempt == 2:
                            turns.append({"name": name, "error": repr(e)[:300]})
                        time.sleep(2)
                print(json.dumps({k: turns[-1].get(k) for k in ("name", "asr_endpoint_s", "first_response_s",
                                  "first_substantive_audio_s", "answer_audio_all_ready_s", "answer_playback_end_s")},
                                 ensure_ascii=False), flush=True)
    finally:
        user_tts.close()
        agent_tts.close()

    ok = [t for t in turns if "error" not in t]

    def stat(key):
        xs = [t[key] for t in ok if t.get(key) is not None]
        return {"p50": pct(xs, 0.5), "p90": pct(xs, 0.9), "max": round(max(xs), 3), "n": len(xs)} if xs else None

    result = {
        "models": {"fast": cloud.FAST_MODEL, "slow": cloud.SLOW_MODEL, "tts": cloud.TTS_MODEL, "asr": "tencent 16k_zh realtime"},
        "asr_vad_silence_ms": 500,
        "tts_speech_rate": SPEECH_RATE,
        "turns_ok": len(ok),
        "turns_error": [t for t in turns if "error" in t],
        "asr_endpoint_s": stat("asr_endpoint_s"),
        "first_response_s": stat("first_response_s"),
        "first_substantive_audio_s": stat("first_substantive_audio_s"),
        "answer_audio_all_ready_s": stat("answer_audio_all_ready_s"),
        "answer_playback_end_s": stat("answer_playback_end_s"),
        "turns": ok,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp4_e2e_v3_voice_cap.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    res = main()
    for k in ("asr_endpoint_s", "first_response_s", "first_substantive_audio_s", "answer_audio_all_ready_s", "answer_playback_end_s"):
        print(k, res[k])
    print("errors", res["turns_error"])
