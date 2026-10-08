"""Live-cloud ASR/preparation/TTS overlap. Playback end is simulated, not a client measurement.

python experiments/exp20_voice_overlap.py --limit 2 --vad-ms 500
python experiments/exp20_voice_overlap.py --vad-ms 300 500 700 --semantic-probe
Semantic decisions are shadow measurements; they never close a turn or release speech.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import queue
import sys
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent import cloud
from agent.pipeline import HybridRetriever, Pipeline, VOICE_RULES
from agent.session import CallSession
from agent.speculative import SpeculativeTurn
from embed_cache import corpus, load
from exp4_e2e import speech_end_s, OPENER

CASES = [
    ("报价", "我三十岁，有社保，一年多少钱？"),
    ("改口", "我三十岁，有社保，哦不对，我没有社保，一年多少钱？"),
    ("等待期", "生病住院和意外住院都要等三十天吗？"),
    ("复合", "住院和门诊怎么赔，ICU还有补贴吗？"),
    ("算术", "急诊花五千，住院花两万，有社保结算，一共能报多少？"),
    ("概览", "这个保险主要保些什么？"),
]
ARMS = {"baseline": (False, False), "stream": (False, True),
        "prepare": (True, False), "prepare_stream": (True, True)}


def semantic_probe(client, text):
    t0 = time.perf_counter()
    answer = cloud.chat([
        {"role": "system", "content": "判断用户这段话在语义上是否说完整了。仅返回完整或未完。条件尚未说完、转折后尚未补充、只有半个问题时返回未完。"},
        {"role": "user", "content": text}], model=cloud.FAST_MODEL, client=client, temperature=0)
    return {"text": text, "decision": answer, "decision_s": time.perf_counter() - t0,
            "finished_wall_s": time.perf_counter()}


def trial(pipe, pcm, tts, opener, arm, vad_ms, semantic, prepare_policy="pause"):
    prepare, stream = ARMS[arm]
    session = CallSession(pipe)
    session.opening()
    session.profile.update(relation="本人", age=30, has_social=True)
    session.flags["quoted"] = True
    manager = SpeculativeTurn(session, streaming=stream,
                              min_interval_s=0.2 if prepare_policy == "pause" else 0.6)
    work = ThreadPoolExecutor(max_workers=1)
    probe_work = ThreadPoolExecutor(max_workers=1)
    audio_queue = queue.Queue()
    synthesized, errors, probes = [], [], []
    origin = [None]
    final = [None]
    future = [None]
    transcript = [None]
    text_done = [None]
    latest_partial = [""]
    quiet_ms = [0]
    probed_pause = [False]
    last_final_wall = [None]

    def audio_worker():
        while True:
            item = audio_queue.get()
            if item is None:
                return
            sentence, emitted = item
            start = time.perf_counter() - origin[0]
            try:
                first, done, audio = tts.synth(sentence)
                synthesized.append({"text": sentence, "emit_s": emitted,
                                    "first_s": start + first, "done_s": start + done,
                                    "duration_s": len(audio) / 2 / 24000})
            except Exception as exc:
                errors.append(type(exc).__name__)

    thread = threading.Thread(target=audio_worker)
    thread.start()

    def emit(sentence):
        audio_queue.put((sentence, time.perf_counter() - origin[0]))

    def confirm(text):
        result = manager.confirm(text, emit)
        text_done[0] = time.perf_counter() - origin[0]
        return result

    def event(ev):
        if origin[0] is None:
            origin[0] = time.perf_counter() - ev.t
        if ev.slice_type in (0, 1) and ev.text and future[0] is None:
            latest_partial[0] = ev.text
            if prepare:
                manager.invalidate(ev.text)
                if prepare_policy == "eager" or quiet_ms[0] >= 120:
                    manager.partial(ev.text)
        elif ev.slice_type == 2 and future[0] is None:
            last_final_wall[0] = time.perf_counter()
            final[0], transcript[0] = ev.t, ev.text
            future[0] = work.submit(confirm, ev.text)

    def audio_frame(frame, audio_t):
        # A small local energy detector starts the model *during* the silence window.
        # This is only a shadow probe, not a production VAD or an endpoint override.
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768
        db = 20 * np.log10(np.sqrt(np.mean(samples ** 2)) + 1e-9)
        if db > -40:
            quiet_ms[0], probed_pause[0] = 0, False
            return
        quiet_ms[0] += len(frame) / 2 / 16
        if prepare and prepare_policy == "pause" and quiet_ms[0] >= 120 and future[0] is None:
            manager.partial(latest_partial[0])
        if (semantic and quiet_ms[0] >= 200 and not probed_pause[0]
                and future[0] is None and len(latest_partial[0]) >= 6 and len(probes) < 3):
            probed_pause[0] = True
            probes.append((audio_t, probe_work.submit(semantic_probe, pipe.client, latest_partial[0])))

    try:
        events, _ = asyncio.run(cloud.tencent_asr_stream(
            pcm, vad_silence_ms=vad_ms, on_event=event, on_audio=audio_frame))
        if future[0] is None:
            raise RuntimeError("ASR produced no final sentence")
        finals = [e for e in events if e.slice_type == 2]
        if len(finals) != 1:
            error = RuntimeError("multiple ASR sentence finals")
            error.voice_diagnostics = {"error_reason": "ASR split one intended turn into multiple final sentences",
                                       "asr_events": [vars(e) for e in events],
                                       "final_count": len(finals),
                                       "semantic_shadow": [{"pause_probe_audio_s": t, **f.result()}
                                                           for t, f in probes]}
            raise error
        result = future[0].result()
        audio_queue.put(None)
        thread.join()
        if errors:
            raise RuntimeError("TTS failed: " + ",".join(errors))
        end = speech_end_s(pcm)
        cursor = final[0] + len(opener) / 2 / 24000
        first_audio = None
        for sent in synthesized:
            cursor = max(cursor, sent["first_s"])
            if first_audio is None:
                first_audio = cursor
            cursor += sent["duration_s"]
        return {"arm": arm, "vad_ms": vad_ms, "prepare_policy": prepare_policy,
                "asr_text": transcript[0], "text": result.text,
                "events": result.events, "preparation": manager.stats, "timings": result.timings,
                "asr_endpoint_s": final[0] - end, "first_response_s": final[0] - end,
                "first_substantive_audio_s": first_audio - end if first_audio is not None else None,
                "text_ready_s": text_done[0] - end,
                "all_audio_ready_s": max((s["done_s"] for s in synthesized), default=text_done[0]) - end,
                "simulated_playback_end_s": cursor - end, "sentences": synthesized,
                "asr_events": [vars(e) for e in events],
                "semantic_shadow": [{"pause_probe_audio_s": t, **f.result(),
                                     "finished_before_asr_final": f.result()["finished_wall_s"] < last_final_wall[0]}
                                    for t, f in probes]}
    finally:
        # Wake the audio worker even when ASR or generation fails.
        work.shutdown(wait=True)
        audio_queue.put(None)
        thread.join()
        manager.close()
        probe_work.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=len(CASES))
    parser.add_argument("--vad-ms", type=int, nargs="+", default=[500])
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--semantic-probe", action="store_true")
    parser.add_argument("--output", default="exp20_voice_overlap.json")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-service-errors", action="store_true")
    parser.add_argument("--answer-provider", choices=["qwen", "deepseek"], default="qwen")
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--prepare-policy", choices=["pause", "eager"], default="pause")
    parser.add_argument("--cases", nargs="+", choices=[c[0] for c in CASES])
    args = parser.parse_args()
    # Report missing configuration without printing credential values.
    import os
    required = ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_WS_URL", "TENCENT_APP_ID",
                "TENCENT_SECRET_ID", "TENCENT_SECRET_KEY")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        parser.error("Missing service configuration: " + ", ".join(missing))
    client = cloud.qwen_client()
    answer_model = args.answer_model or cloud.QWEN_ANSWER_MODEL
    if args.answer_provider == "deepseek":
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env.deepseek")
        if not os.getenv("DEEPSEEK_API_KEY"):
            parser.error("Missing service configuration: DEEPSEEK_API_KEY")
        from model_router import AnswerModelRouter
        answer_model = args.answer_model or "deepseek-flash"
        client = AnswerModelRouter(client, answer_model)
    chunks = corpus(False)
    pipe = Pipeline(HybridRetriever(chunks, load(chunks, "clauses"), client=client), client=client,
                    slow_model=answer_model)
    user_tts = cloud.RealtimeTTS(voice="Ethan")
    agent_tts = cloud.RealtimeTTS(voice="Cherry", speech_rate=1.2)
    path = ROOT / "experiments/results" / Path(args.output).name
    previous = json.loads(path.read_text(encoding="utf-8")) if args.resume and path.exists() else {}
    rows = previous.get('rows', [])
    previous_errors = previous.get('previous_service_errors', [])
    if args.retry_service_errors:
        failed = [r for r in rows if 'error' in r and not r.get('error_reason')]
        previous_errors += failed
        rows = [r for r in rows if r not in failed]
    remaining_done = {}
    for row in rows:
        key = (row["case"], row["vad_ms"], row["arm"])
        remaining_done[key] = remaining_done.get(key, 0) + 1
    try:
        _, _, opener = agent_tts.synth(OPENER)
        selected = [case for case in CASES[:args.limit] if not args.cases or case[0] in args.cases]
        for name, question in selected * args.rounds:
            audio_path = ROOT / "experiments/recordings/exp20_inputs" / (hashlib.sha256(question.encode()).hexdigest()[:16] + ".pcm")
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            if audio_path.exists():
                pcm = audio_path.read_bytes()
            else:
                _, _, audio = user_tts.synth(question)
                pcm = cloud.resample_24k_to_16k(audio)
                audio_path.write_bytes(pcm)
            for vad in args.vad_ms:
                for arm in args.arms:
                    key = (name, vad, arm)
                    if remaining_done.get(key, 0):
                        remaining_done[key] -= 1
                        continue
                    try:
                        row = {"case": name, "question": question,
                               **trial(pipe, pcm, agent_tts, opener, arm, vad, args.semantic_probe, args.prepare_policy)}
                    except Exception as exc:
                        row = {"case": name, "arm": arm, "vad_ms": vad, "error": type(exc).__name__,
                               **getattr(exc, "voice_diagnostics", {})}
                    rows.append(row)
                    payload = json.dumps({"voice_rules": VOICE_RULES, "playback": "simulated",
                                                "answer_provider": args.answer_provider, "answer_model": answer_model,
                                                "fast_model": cloud.FAST_MODEL,
                                                "quality": "requires human review; no accuracy claim",
                                                "previous_service_errors": previous_errors,
                                                "rows": rows}, ensure_ascii=False, indent=2)
                    for attempt in range(10):
                        try:
                            temp = path.with_suffix('.tmp')
                            temp.write_text(payload, encoding='utf-8')
                            temp.replace(path)
                            break
                        except OSError:
                            if attempt == 9:
                                raise
                            time.sleep(.1)
                    print(json.dumps({k: row.get(k) for k in ("case", "arm", "vad_ms", "error", "preparation",
                          "first_substantive_audio_s", "simulated_playback_end_s")}, ensure_ascii=False), flush=True)
    finally:
        user_tts.close()
        agent_tts.close()


if __name__ == "__main__":
    main()
