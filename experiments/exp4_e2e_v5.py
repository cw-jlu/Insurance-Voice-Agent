"""实验 4 第八轮重测：端到端语音时延（与第六轮 exp4_e2e_v4 相同的 12 类问题 × 2 轮、相同的测量口径）。

与第六轮的区别（第八轮的时延改动）：
- 检索用用户原话，与理解、对话作用、缺口检测四路并行（exp13：原话命中率反而更高）；
- 报价、等待期、续保/清单类缺口由程序直接生成，不调用大模型；
- 复合问题不再走 Agent 叠 step，理解模块一次最多选 5 项责任（exp14）；
- 事实核对默认关闭（第七轮：拦下错误 4/13、误拦正确 10/67），缺口复核与事实核对（若开启）并行；
- 语音合成：3 条持久连接同时合成一段回答里的各句；流程话术、模板等固定句子在通话前合成好，播放时不再合成。
时延从用户说完话算起；"回答音频全部合成完" = 每句合成完成时刻的最大值（并行后不再逐句累加）。

用法：python experiments/exp4_e2e_v5.py [tag]
"""

from __future__ import annotations

import asyncio
import json
import re
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
from agent.session import CallSession, fixed_phrases  # noqa: E402
from embed_cache import corpus, load  # noqa: E402
from exp4_e2e import OPENER, SPEECH_RATE, speech_end_s, to24k  # noqa: E402
from exp4_e2e_v4 import UTTERANCES, pct  # noqa: E402

OUT = ROOT / "experiments" / "results"
REC = ROOT / "experiments" / "recordings"
SR = 24000
TAG = sys.argv[1] if len(sys.argv) > 1 else ""


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[。！？?])", text) if s.strip()]


def run_turn(name, text, known, pipe, user_tts, pool, opener_pcm, rec_path):
    _, _, user24 = user_tts.synth(text)
    user16 = cloud.resample_24k_to_16k(user24)
    t_end = speech_end_s(user16)
    events, _ = asyncio.run(cloud.tencent_asr_stream(user16, vad_silence_ms=500))
    finals = [e for e in events if e.slice_type == 2]
    if not finals:
        raise RuntimeError("no final ASR result")
    t_final = finals[-1].t
    asr_text = "".join(e.text for e in finals)

    s = CallSession(pipe, agent_steps=0)  # exp14: evidence agent replaced by understanding with up to 5 items
    s.opening()
    s.profile.update(known)
    if known.get("age") is not None:
        s.flags["quoted"] = True  # mid-call: the price was already given earlier

    track = [(t_final, opener_pcm, OPENER)]
    cursor = t_final + len(opener_pcm) / 2 / SR
    t0 = time.perf_counter()
    r = s.turn(asr_text)
    t_text = t_final + (time.perf_counter() - t0)

    sentences = split_sentences(r.text)
    synth = pool.synth_all(sentences)  # concurrent; cached sentences cost nothing
    first_audio, all_ready, cached = None, t_text, 0
    for sent, (first, done, pcm, hit) in zip(sentences, synth):
        cached += hit
        start = max(cursor, t_text + first)
        first_audio = first_audio if first_audio is not None else start
        track.append((start, pcm, sent))
        all_ready = max(all_ready, t_text + done)
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
        "name": name, "asr_text": asr_text, "events": r.events, "stage_s": {k: round(v, 3) for k, v in r.timings.items() if isinstance(v, float)},
        "asr_endpoint_s": round(t_final - t_end, 3),
        "first_response_s": round(t_final - t_end, 3),
        "answer_text_ready_s": round(t_text - t_end, 3),
        "first_substantive_audio_s": round(first_audio - t_end, 3) if first_audio else None,
        "answer_audio_all_ready_s": round(all_ready - t_end, 3),
        "answer_playback_end_s": round(cursor - t_end, 3),
        "agent_steps": next((int(e.split("Agent ")[1].split("步")[0]) for e in r.events if "Agent " in e), 0),
        "answer_chars": len(r.text), "sentences": len(sentences), "cached_sentences": cached,
        "text": r.text, "recording": str(rec_path.relative_to(ROOT)),
    }


def main(rounds: int = 2, presynth_fixed: bool = True):
    client = cloud.llm_client()
    chunks = corpus(False)
    pipe = Pipeline(HybridRetriever(chunks, load(chunks, "clauses"), client=client), client=client)
    user_tts = cloud.RealtimeTTS(voice="Ethan")
    pool = cloud.TTSPool(3, voice="Cherry", speech_rate=SPEECH_RATE)
    t0 = time.perf_counter()
    pool.presynth(fixed_phrases() + [OPENER] if presynth_fixed else [OPENER])
    presynth_s = time.perf_counter() - t0
    opener_pcm = pool.cache[OPENER]
    turns = []
    try:
        for r in range(rounds):
            for name, text, known in UTTERANCES:
                for attempt in range(3):
                    try:
                        turns.append(run_turn(name, text, known, pipe, user_tts, pool, opener_pcm,
                                              REC / f"v5{TAG}_r{r + 1}_{name}.wav"))
                        break
                    except Exception as e:
                        if attempt == 2:
                            turns.append({"name": name, "error": repr(e)[:200]})
                        time.sleep(2)
                t = turns[-1]
                print(json.dumps({k: t.get(k) for k in ("name", "answer_text_ready_s", "first_substantive_audio_s",
                                  "answer_audio_all_ready_s", "answer_playback_end_s", "agent_steps", "cached_sentences",
                                  "sentences")}, ensure_ascii=False), flush=True)
    finally:
        user_tts.close()
        pool.close()

    ok = [t for t in turns if "error" not in t]

    def stat(key, sel=lambda t: True):
        xs = [t[key] for t in ok if sel(t) and t.get(key) is not None]
        return {"p50": pct(xs, 0.5), "p90": pct(xs, 0.9), "max": round(max(xs), 3), "n": len(xs)} if xs else None

    keys = ("first_response_s", "answer_text_ready_s", "first_substantive_audio_s", "answer_audio_all_ready_s", "answer_playback_end_s")
    res = {
        "version": "第八轮（四路并行理解+原话检索、模板答案、事实核对关闭、TTS 3 连接并行+固定句预合成）",
        "presynth_fixed": presynth_fixed,
        "presynth_s": round(presynth_s, 1), "presynth_sentences": len(pool.cache),
        "turns_ok": len(ok), "errors": [t for t in turns if "error" in t],
        "all": {k: stat(k) for k in keys},
        "simple": {k: stat(k, lambda t: not t["name"].startswith("复合")) for k in keys},
        "complex": {k: stat(k, lambda t: t["name"].startswith("复合")) for k in keys},
        "over_10s_all_ready": sum(1 for t in ok if t["answer_audio_all_ready_s"] > 10),
        "turns": ok,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"exp4_e2e_v5{TAG}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main()
    for grp in ("all", "simple", "complex"):
        print(grp, json.dumps(r[grp], ensure_ascii=False))
    print("over 10s (all audio ready):", r["over_10s_all_ready"], "errors:", len(r["errors"]))
