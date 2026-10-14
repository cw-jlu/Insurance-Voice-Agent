"""Use streaming ASR as a limited transcription check for speech-rate samples."""

from __future__ import annotations

import asyncio
import json
import re
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.cloud import resample_24k_to_16k, tencent_asr_stream  # noqa: E402

RESULT = ROOT / "experiments/results/exp12_speech_rate.json"
RATES = (1.15, 1.5, 2.0)


def normalize(text: str) -> str:
    return "".join(re.findall(r"[\w]+", text.lower()))


def distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1]


def main():
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    source = json.loads((ROOT / data["source"]).read_text(encoding="utf-8"))
    case = next(t for t in source["turns"] if t["name"] == "复合4")
    reference = normalize(case["first_phrase"] + case["text"])
    checks = []
    for rate in RATES:
        path = ROOT / "experiments/recordings" / f"speech_rate_{str(rate).replace('.', 'p')}_complex4.wav"
        with wave.open(str(path), "rb") as wav:
            pcm = wav.readframes(wav.getnframes())
        events, _ = asyncio.run(tencent_asr_stream(resample_24k_to_16k(pcm)))
        transcript = "".join(e.text for e in events if e.slice_type == 2)
        observed = normalize(transcript)
        checks.append({
            "rate": rate,
            "reference": case["first_phrase"] + case["text"],
            "asr_transcript": transcript,
            "normalized_edit_distance": round(distance(reference, observed) / len(reference), 3),
            "audio_s": round(len(pcm) / 48000, 3),
        })
        print(json.dumps(checks[-1], ensure_ascii=False), flush=True)
    data["asr_check_complex4"] = checks
    RESULT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
