"""实验 4（第十轮）：大链路接入流式输出后重测端到端语音时延。

与 exp4_e2e_v5.py 用例、计时口径相同（30 句 × 2 轮），区别只在回答的交付方式：
- 原来：CallSession.turn 返回整段文字后才开始逐句合成；
- 现在：CallSession.turn(on_sentence=...) 每出一句（过规则护栏后）就放进队列，后台线程立即合成，
  合成与模型生成、与播放三者重叠。
时延仍从用户说完话算起；"回答文字生成完"= turn 返回的时刻。事实核对保持默认关闭。
"""

from __future__ import annotations

import asyncio
import queue
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
from agent.session import CallSession  # noqa: E402
import exp4_e2e_v4 as v4  # noqa: E402
import exp4_e2e_v5 as v5  # noqa: E402
from exp4_e2e import speech_end_s, to24k  # noqa: E402

SR = v4.SR


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
        s.flags["quoted"] = True

    # Wall clock origin = moment the ASR closed the sentence (t_final on the audio timeline).
    w0 = time.perf_counter()
    q: queue.Queue = queue.Queue()
    synthesized = []  # (sentence, emit_t, synth_start_t, first_audio_t, done_t, pcm) on the audio timeline
    errors = []

    def worker():
        while True:
            item = q.get()
            if item is None:
                return
            sent, emit_t = item
            try:
                start = t_final + (time.perf_counter() - w0)
                f, tot, pcm = agent_tts.synth(sent)
                synthesized.append((sent, emit_t, start, start + f, start + tot, pcm))
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e)[:200])

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    r = s.turn(asr_text, on_sentence=lambda x: q.put((x, t_final + (time.perf_counter() - w0))))
    t_text = t_final + (time.perf_counter() - w0)
    q.put(None)
    th.join()
    if errors:
        raise RuntimeError(errors[0])

    track = [(t_final, opener_pcm, v4.OPENER)]
    cursor = t_final + len(opener_pcm) / 2 / SR
    first_audio = None
    for sent, _, _, first_t, _, pcm in synthesized:
        start = max(cursor, first_t)
        first_audio = first_audio if first_audio is not None else start
        track.append((start, pcm, sent))
        cursor = start + len(pcm) / 2 / SR
    all_ready = max((x[4] for x in synthesized), default=t_text)

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
        "first_sentence_emitted_s": round(synthesized[0][1] - t_end, 3) if synthesized else None,
        "answer_text_ready_s": round(t_text - t_end, 3),
        "first_substantive_audio_s": round(first_audio - t_end, 3) if first_audio else None,
        "answer_audio_all_ready_s": round(all_ready - t_end, 3),
        "answer_playback_end_s": round(cursor - t_end, 3),
        "agent_steps": next((int(e.split("Agent ")[1].split("步")[0]) for e in r.events if "Agent " in e), 0),
        "sentences": len(synthesized),
        "answer_chars": len(r.text),
        "text": r.text,
        "tts_reconnects": 0,
        "recording": str(rec_path.relative_to(ROOT)),
    }


if __name__ == "__main__":
    v4.run_turn = run_turn
    v4.UTTERANCES = v4.UTTERANCES + v5.NEW
    v4.TAG = sys.argv[1] if len(sys.argv) > 1 else "_stream"
    r = v4.main(rounds=2)
    for grp in ("all", "simple", "complex"):
        print(grp, r[grp])
    print("over 10s (all audio ready):", r["over_10s_all_ready"], "/", r["turns_ok"], "errors:", len(r["errors"]))
