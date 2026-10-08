"""Summarize saved trials without sending data to a scoring model."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--answers", action="store_true")
    args = parser.parse_args()
    for path in args.files:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = data["rows"]
        print(Path(path).name)
        for arm in ("baseline", "stream", "prepare", "prepare_stream"):
            group = [r for r in rows if r["arm"] == arm]
            good = [r for r in group if "error" not in r]
            generated = [r for r in good if r.get("timings", {}).get("answer_detail", {}).get("generate_1") is not None]
            def median(key, samples):
                values = [r[key] for r in samples if r.get(key) is not None]
                return round(statistics.median(values), 3) if values else None
            print(json.dumps({"arm": arm, "valid": len(good), "errors": len(group)-len(good),
                              "generated": len(generated),
                              "first_audio_p50": median("first_substantive_audio_s", good),
                              "generated_first_audio_p50": median("first_substantive_audio_s", generated),
                              "generated_playback_p50": median("simulated_playback_end_s", generated),
                              "over_10s_playback": sum(r["simulated_playback_end_s"] > 10 for r in good),
                              "reused": sum(r["preparation"]["reused"] for r in good)}, ensure_ascii=False))
        if args.answers:
            for row in rows:
                print(json.dumps({k: row.get(k) for k in ("case", "arm", "text", "error", "error_reason")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
