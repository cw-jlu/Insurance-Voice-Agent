"""实验 4 重测：用当前版本（第六轮）测端到端语音时延。

与第三轮的区别：回答走大链路 CallSession.turn（理解 + 对话作用 + 缺口检测并行、证据分组、护栏 + 小模型事实核对、
复合问题 Agent 叠 step、门禁与流程追问），不再是只做单轮生成的流式版本。
每轮：TTS 合成用户语音 → 按实时速度推给腾讯云 ASR → 整句结果出来后立即播放预合成承接语 →
CallSession.turn 生成回答 → 持久连接 TTS 逐句合成 → 按真实时间排进双声道录音。时延从用户说完话算起。
除第三轮的 8 类问题外，新增 4 道复合问题（会触发 Agent）。
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agent import cloud  # noqa: E402
from agent.pipeline import HybridRetriever, Pipeline  # noqa: E402
from agent.session import CallSession  # noqa: E402
from embed_cache import corpus, load  # noqa: E402
from exp4_e2e import OPENER, SPEECH_RATE, speech_end_s, to24k  # noqa: E402

OUT = ROOT / "experiments" / "results"
REC = ROOT / "experiments" / "recordings"
SR = 24000
TAG = sys.argv[1] if len(sys.argv) > 1 else ""

# (名称, 用户这句话, 通话里此前已知的条件)
UTTERANCES = [
    ("报价", "我三十岁，有社保，一年多少钱？", {"relation": "本人"}),
    ("病史", "我有高血压好几年了，一直在吃降压药，买这个以后住院能报吗？", {"relation": "本人", "age": 50, "has_social": True}),
    ("免赔额", "我自己先掏一万，剩下的是不是全报？", {"relation": "本人", "age": 30, "has_social": True}),
    ("保障范围", "刷到广告说几百万保额，具体都能报什么？", {}),
    ("价格变化", "以后每年都是这个价吗？", {"relation": "本人", "age": 30, "has_social": True}),
    ("身份", "你是真人还是机器人？", {}),
    ("等待期", "等待期是多久？", {"relation": "本人", "age": 30, "has_social": True}),
    ("加油包", "门急诊加油包多大年纪能买？", {"relation": "本人", "age": 30, "has_social": True}),
    ("复合1", "住院和看门诊分别怎么报，进ICU有没有补贴？", {"relation": "本人", "age": 40, "has_social": True}),
    ("复合2", "得了癌症，住院、靶向药和质子重离子都能报吗，有免赔额吗？", {"relation": "本人", "age": 40, "has_social": True}),
    ("复合3", "我女儿8岁，近视手术和看牙能报吗，住院呢？", {"relation": "子女", "age": 8, "has_social": True}),
    ("复合4", "急诊抢救花了五千，后来住院花了两万，一共能报多少？", {"relation": "本人", "age": 40, "has_social": True}),
]


def run_turn(name, text, known, pipe, user_tts, agent_tts, opener_pcm, rec_path):
    _, _, user24 = user_tts.synth(text)
    user16 = cloud.resample_24k_to_16k(user24)
    t_end = speech_end_s(user16)
    events, _ = asyncio.run(cloud.tencent_asr_stream(user16, vad_silence_ms=500))
    finals = [e for e in events if e.slice_type == 2]
    if not finals:
        raise RuntimeError("no final ASR result")
    t_final = finals[-1].t
    asr_text = "".join(e.text for e in finals)

    s = CallSession(pipe, agent_steps=5)
    s.opening()
    s.profile.update(known)
    if known.get("age") is not None:
        s.flags["quoted"] = True  # mid-call: the price was already given earlier

    track = [(t_final, opener_pcm, OPENER)]  # pre-synthesized opener plays as soon as ASR closes the sentence
    cursor = t_final + len(opener_pcm) / 2 / SR
    t0 = time.perf_counter()
    r = s.turn(asr_text)
    t_text = t_final + (time.perf_counter() - t0)

    sentences = [x for x in r.text.replace("？", "？\n").replace("。", "。\n").split("\n") if x.strip()]
    first_audio, all_ready = None, t_text
    for sent in sentences:
        f, tot, pcm = agent_tts.synth(sent)
        ready_first = all_ready + f
        start = max(cursor, ready_first)
        first_audio = first_audio if first_audio is not None else start
        track.append((start, pcm, sent))
        all_ready += tot
        cursor = start + len(pcm) / 2 / SR

    total = max(len(user16) / 2 / 16000, cursor) + 0.5
    left = np.zeros(int(total * SR), dtype=np.float32)
    right = np.zeros_like(left)
    u = to24k(user16)
    left[: len(u)] = u[: len(left)]
    for start, pcm, _ in track:
        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        i = int(start * SR)
        right[i : i + len(a)] += a[: max(0, len(right) - i)]
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(rec_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.stack([left, right], axis=1).clip(-32768, 32767).astype(np.int16).tobytes())

    return {
        "name": name, "asr_text": asr_text, "events": r.events,
        "asr_endpoint_s": round(t_final - t_end, 3),
        "first_response_s": round(t_final - t_end, 3),
        "answer_text_ready_s": round(t_text - t_end, 3),
        "first_substantive_audio_s": round(first_audio - t_end, 3) if first_audio else None,
        "answer_audio_all_ready_s": round(all_ready - t_end, 3),
        "answer_playback_end_s": round(cursor - t_end, 3),
        "agent_steps": next((int(e.split("Agent ")[1].split("步")[0]) for e in r.events if "Agent " in e), 0),
        "answer_chars": len(r.text),
        "text": r.text,
        "recording": str(rec_path.relative_to(ROOT)),
    }


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))], 3)


def main(rounds: int = 2):
    client = cloud.llm_client()
    chunks = corpus(False)
    pipe = Pipeline(HybridRetriever(chunks, load(chunks, "clauses"), client=client), client=client)
    user_tts = cloud.RealtimeTTS(voice="Ethan")
    agent_tts = cloud.RealtimeTTS(voice="Cherry", speech_rate=SPEECH_RATE)
    _, _, opener_pcm = agent_tts.synth(OPENER)
    turns = []
    try:
        for r in range(rounds):
            for name, text, known in UTTERANCES:
                for attempt in range(3):
                    try:
                        turns.append(run_turn(name, text, known, pipe, user_tts, agent_tts, opener_pcm,
                                              REC / f"v4{TAG}_r{r + 1}_{name}.wav"))
                        break
                    except Exception as e:
                        if attempt == 2:
                            turns.append({"name": name, "error": repr(e)[:200]})
                        time.sleep(2)
                t = turns[-1]
                print(json.dumps({k: t.get(k) for k in ("name", "first_response_s", "answer_text_ready_s",
                                  "first_substantive_audio_s", "answer_audio_all_ready_s", "answer_playback_end_s",
                                  "agent_steps")}, ensure_ascii=False), flush=True)
    finally:
        user_tts.close()
        agent_tts.close()

    ok = [t for t in turns if "error" not in t]

    def stat(key, sel=lambda t: True):
        xs = [t[key] for t in ok if sel(t) and t.get(key) is not None]
        return {"p50": pct(xs, 0.5), "p90": pct(xs, 0.9), "max": round(max(xs), 3), "n": len(xs)} if xs else None

    keys = ("first_response_s", "answer_text_ready_s", "first_substantive_audio_s", "answer_audio_all_ready_s", "answer_playback_end_s")
    res = {
        "version": "第六轮（CallSession + 事实核对 + 复合问题 Agent）",
        "turns_ok": len(ok), "errors": [t for t in turns if "error" in t],
        "all": {k: stat(k) for k in keys},
        "simple": {k: stat(k, lambda t: not t["name"].startswith("复合")) for k in keys},
        "complex": {k: stat(k, lambda t: t["name"].startswith("复合")) for k in keys},
        "over_10s_all_ready": sum(1 for t in ok if t["answer_audio_all_ready_s"] > 10),
        "turns": ok,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"exp4_e2e_v4{TAG}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main()
    for grp in ("all", "simple", "complex"):
        print(grp, json.dumps(r[grp], ensure_ascii=False))
    print("over 10s (all audio ready):", r["over_10s_all_ready"], "errors:", len(r["errors"]))
