"""实验 5（真实音频）：播报中插话。

麦克风信号 = 用户语音 + Agent 播报的泄漏（回声）：
- "有回声消除"：泄漏为 Agent 音量的 3%；
- "无回声消除"：泄漏为 30%（外放、无 AEC 的最坏情况）。
把混合信号按实时速度推给腾讯云实时 ASR，用 BargeInController 判断。
三类用例：真打断、附和、只有回声（用户不说话）。
停播时延从用户开口时刻算起。用户语音为 TTS 合成，比真人干净。
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agent import cloud, small_model  # noqa: E402
from agent.dialogue import BargeInController  # noqa: E402
from exp4_e2e import speech_end_s  # noqa: E402

OUT = ROOT / "experiments" / "results"
rng = random.Random(20261009)

AGENT = ["您30岁有社保的话，基础版一年318元，按月交每月31元，一年下来年缴更划算一些。",
         "一般住院一年累计超过1万的部分，符合条件的按100%报，但要在二级及以上公立医院普通部看病。",
         "高血压属于既往症，相关的住院费用是不赔的，能不能投保要先做健康告知，以核保结果为准。",
         "重大疾病医疗没有1万免赔额，按100%报，一年最多300万，还包括住院前后的门急诊费用。"]
INTERRUPTS = ["等一下，我没有社保", "不对，我是给我妈买的", "那按月交是多少", "我不需要了", "你是机器人吗",
              "等等，我35岁", "免赔额是多少", "先停一下"]
BACKCHANNELS = ["嗯", "对", "好的", "嗯嗯，你说", "明白", "哦，然后呢"]


def first_voiced_s(pcm16k: bytes, thresh_db: float = -40.0) -> float:
    x = np.frombuffer(pcm16k, dtype=np.int16).astype(np.float32) / 32768
    n = 320
    frames = x[: len(x) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    v = np.where(db > thresh_db)[0]
    return v[0] * 0.02 if len(v) else 0.0


def mix(agent16: bytes, user16: bytes | None, user_at_s: float, echo_gain: float) -> tuple[bytes, float, float]:
    a = np.frombuffer(agent16, dtype=np.int16).astype(np.float32) * echo_gain
    delay = int(0.05 * 16000)
    a = np.concatenate([np.zeros(delay, dtype=np.float32), a])
    total = len(a) + 16000 * 3
    m = np.zeros(total, dtype=np.float32)
    m[: len(a)] += a
    onset = None
    if user16 is not None:
        u = np.frombuffer(user16, dtype=np.int16).astype(np.float32)
        i = int(user_at_s * 16000)
        m[i : i + len(u)] += u[: total - i]
        onset = user_at_s + first_voiced_s(user16)
    m += np.random.default_rng(1).normal(0, 60, total).astype(np.float32)  # room noise floor
    agent_end = len(a) / 16000
    return np.clip(m, -32768, 32767).astype(np.int16).tobytes(), onset, agent_end


async def trial(kind: str, text: str | None, agent_pcm: bytes, agent_text: str, user_pcm: bytes | None,
                echo_gain: float, client) -> dict:
    user_at = rng.uniform(1.0, 2.5)
    mic, onset, agent_end = mix(agent_pcm, user_pcm, user_at, echo_gain)
    ctrl = BargeInController(lambda t: small_model.classify_barge_in(t, client=client))
    ctrl.note_spoken(0.0, agent_text)
    events, _ = await cloud.tencent_asr_stream(mic, vad_silence_ms=500, trailing_silence_ms=0)
    decisions = []
    stopped_at = None
    for e in events:
        if stopped_at is not None:
            break
        speaking = e.t < agent_end
        t0 = time.perf_counter()
        d = ctrl.on_partial(e.t, e.text, speaking)
        if d is None:
            continue
        d_t = e.t + (time.perf_counter() - t0)  # include classification time
        decisions.append({"t": round(d_t, 3), "action": d.action, "reason": d.reason, "asr": e.text})
        if d.action == "stop":
            stopped_at = d_t
    return {"kind": kind, "user": text, "agent": agent_text[:20], "echo_gain": echo_gain,
            "user_onset_s": round(onset, 3) if onset else None, "stopped": stopped_at is not None,
            "stop_latency_s": round(stopped_at - onset, 3) if (stopped_at and onset) else None,
            "decisions": decisions[:6]}


async def main_async():
    client = cloud.llm_client()
    agent_tts = cloud.RealtimeTTS(voice="Cherry")
    user_tts = cloud.RealtimeTTS(voice="Ethan")
    agent_audio = {s: cloud.resample_24k_to_16k(agent_tts.synth(s)[2]) for s in AGENT}
    user_audio = {u: cloud.resample_24k_to_16k(user_tts.synth(u)[2]) for u in INTERRUPTS + BACKCHANNELS}
    agent_tts.close()
    user_tts.close()
    jobs = []
    for gain, cond in ((0.03, "有回声消除"), (0.3, "无回声消除")):
        for u in INTERRUPTS:
            a = rng.choice(AGENT)
            jobs.append((cond, "打断", u, a, user_audio[u], gain))
        for u in BACKCHANNELS:
            a = rng.choice(AGENT)
            jobs.append((cond, "附和", u, a, user_audio[u], gain))
        for a in AGENT:
            jobs.append((cond, "只有回声", None, a, None, gain))
    results = []
    sem = asyncio.Semaphore(4)

    async def run(job):
        cond, kind, u, a, upcm, gain = job
        async with sem:
            for attempt in range(3):
                try:
                    r = await trial(kind, u, agent_audio[a], a, upcm, gain, client)
                    r["condition"] = cond
                    return r
                except Exception as e:
                    if attempt == 2:
                        return {"condition": cond, "kind": kind, "user": u, "error": repr(e)[:200]}
                    await asyncio.sleep(2)

    results = await asyncio.gather(*(run(j) for j in jobs))
    return results


def main():
    rows = asyncio.run(main_async())
    summary = {}
    for cond in ("有回声消除", "无回声消除"):
        rs = [r for r in rows if r.get("condition") == cond and "error" not in r]
        it = [r for r in rs if r["kind"] == "打断"]
        bc = [r for r in rs if r["kind"] == "附和"]
        eo = [r for r in rs if r["kind"] == "只有回声"]
        lat = sorted(r["stop_latency_s"] for r in it if r["stop_latency_s"] is not None)
        summary[cond] = {
            "interrupt_stopped": f"{sum(r['stopped'] for r in it)}/{len(it)}",
            "stop_latency_s": {"p50": round(statistics.median(lat), 3), "max": round(max(lat), 3)} if lat else None,
            "backchannel_wrongly_stopped": f"{sum(r['stopped'] for r in bc)}/{len(bc)}",
            "echo_only_wrongly_stopped": f"{sum(r['stopped'] for r in eo)}/{len(eo)}",
        }
    res = {"summary": summary, "errors": [r for r in rows if "error" in r], "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp5_audio.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps(r["summary"], ensure_ascii=False, indent=1))
    print("errors", len(r["errors"]))
    for row in r["rows"]:
        if "error" in row:
            continue
        bad = (row["kind"] == "打断" and not row["stopped"]) or (row["kind"] != "打断" and row["stopped"])
        if bad:
            print("  ✗", row["condition"], row["kind"], row["user"], row["decisions"][:3])
