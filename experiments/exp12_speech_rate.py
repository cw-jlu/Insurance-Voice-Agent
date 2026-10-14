"""Replay frozen agent text through TTS at different rates and estimate playback end.

The ASR, first-response decision, and answer generation timings come from an
existing full-chain run. Only TTS is rerun, so rate comparisons use identical
words while still measuring actual synthesized audio length and TTS latency.
"""

from __future__ import annotations

import json
import statistics
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.cloud import RealtimeTTS  # noqa: E402

SOURCE = ROOT / "experiments/results/exp4_e2e_v4_new2.json"
OUTPUT = ROOT / "experiments/results/exp12_speech_rate.json"
SAMPLES = ROOT / "experiments/recordings"
RATES = (1.15, 1.35, 1.5, 1.8, 2.0)


def sentences(text: str) -> list[str]:
    return [x for x in text.replace("？", "？\n").replace("。", "。\n").split("\n") if x.strip()]


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    return round(values[min(len(values) - 1, int(round(p * (len(values) - 1))))], 3)


def save_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)


def measure(rate: float, turns: list[dict]) -> dict:
    tts = RealtimeTTS(voice="Cherry", speech_rate=rate)
    outputs = []
    try:
        openers = {phrase: tts.synth(phrase)[2] for phrase in {t["first_phrase"] for t in turns}}
        for turn in turns:
            cursor = turn["first_response_s"] + len(openers[turn["first_phrase"]]) / 48000
            all_ready = turn["answer_text_ready_s"]
            first_substantive = None
            audio = bytearray(openers[turn["first_phrase"]])
            for sentence in sentences(turn["text"]):
                first, total, pcm = tts.synth(sentence)
                start = max(cursor, all_ready + first)
                if first_substantive is None:
                    first_substantive = start
                cursor = start + len(pcm) / 48000
                all_ready += total
                audio.extend(pcm)
            if turn["name"] == "复合4":
                save_wav(SAMPLES / f"speech_rate_{str(rate).replace('.', 'p')}_complex4.wav", bytes(audio))
            outputs.append({
                "name": turn["name"],
                "answer_chars": turn["answer_chars"],
                "answer_playback_end_s": round(cursor, 3),
                "first_substantive_audio_s": round(first_substantive, 3),
                "answer_audio_all_ready_s": round(all_ready, 3),
                "spoken_audio_duration_s": round(len(audio) / 48000, 3),
            })
            print(json.dumps({"rate": rate, **outputs[-1]}, ensure_ascii=False), flush=True)
    finally:
        tts.close()
    ends = [t["answer_playback_end_s"] for t in outputs]
    return {
        "rate": rate,
        "p50_end_s": pct(ends, 0.5),
        "p90_end_s": pct(ends, 0.9),
        "max_end_s": max(ends),
        "over_10s": sum(end > 10 for end in ends),
        "mean_audio_s": round(statistics.mean(t["spoken_audio_duration_s"] for t in outputs), 3),
        "turns": outputs,
    }


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "method": "Frozen answer text and upstream timings; actual cloud TTS per rate; simulated sequential playback using the full-chain scheduling logic.",
        "baseline_full_chain": {
            "rate": 1.15,
            "p50_end_s": source["all"]["answer_playback_end_s"]["p50"],
            "p90_end_s": source["all"]["answer_playback_end_s"]["p90"],
            "max_end_s": source["all"]["answer_playback_end_s"]["max"],
            "over_10s": sum(t["answer_playback_end_s"] > 10 for t in source["turns"]),
        },
        "rates": [],
    }
    for rate in RATES:
        try:
            result["rates"].append(measure(rate, source["turns"]))
        except Exception as exc:
            result["rates"].append({"rate": rate, "error": repr(exc)})
            print(f"rate {rate}: {exc!r}", flush=True)
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in result["rates"]:
        print(json.dumps({k: v for k, v in item.items() if k != "turns"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
