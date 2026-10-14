"""Evaluate bounded first replies, link consent, and one-pass answers.

The 80-question set is a regression set after iterative fixes. Use a new dataset argument
for separate generalization checks; the supplied fresh sets are retained with their first-run results.
"""

from __future__ import annotations

import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from agent import cloud, first_response, small_model  # noqa: E402
from agent.pipeline import HybridRetriever, Pipeline, understand  # noqa: E402
from embed_cache import corpus, load  # noqa: E402
from exp2_qa_v2 import model_judge, regex_judge  # noqa: E402

CONSENT_CASES = [
    ("发吧", True, True), ("好的，发我", True, True), ("可以发", True, True),
    ("嗯", True, False), ("把投保链接发给我", False, True),
    ("嗯……那我再考虑考虑吧，先挂了", True, False),
    ("好，我跟家人商量一下", True, False), ("发个介绍给我看看", True, False),
    ("我想买", False, False), ("怎么买", False, False),
    ("先不用发", True, False), ("不用了，先挂了", True, False),
    ("链接先别发，我想想", True, False), ("以后再发吧", True, False),
    ("把投保链接发我邮箱", True, False),
    ("发吧，不过我还得考虑一下", True, False),
    ("那你给我发投保链接", False, True),
]

FIRST_CASES = [
    ("我三十岁，有社保，一年多少钱？", "价格"),
    ("按月交一个月多少？", "价格"),
    ("住院能报什么？", "保障"),
    ("等待期多久？", "保障"),
    ("我有高血压，能买不？", "病史"),
    ("我妈以前做过手术，买了能报吗？", "病史"),
    ("你是真人还是机器人？", "身份"),
    ("给我转人工", "人工"), ("你们别再打了", "人工"),
    ("我自己买", "其他"), ("嗯，好", "其他"),
]


def first_and_consent(client):
    def first(row):
        q, expected = row
        phrase, path, sec = first_response.choose(q, client=client)
        got = next(k for k, v in first_response.PHRASES.items() if v == phrase)
        return {"text": q, "expected": expected, "got": got, "path": path, "seconds": round(sec, 3)}

    def consent(row):
        q, offered, expected = row
        got = small_model.link_consent(q, offered, client=client)
        return {"text": q, "offered": offered, "expected": expected, "got": got}

    # A phone call has one first response at a time; queueing ten calls behind a shared pool
    # would measure test contention rather than the per-turn deadline.
    first_rows = [first(row) for row in FIRST_CASES]
    with ThreadPoolExecutor(max_workers=4) as ex:
        consent_rows = list(ex.map(consent, CONSENT_CASES))
    return {"first": first_rows, "consent": consent_rows}


def qa(client, data_name="qa_v3.json"):
    chunks = corpus(False)
    pipe = Pipeline(HybridRetriever(chunks, load(chunks, "clauses"), client=client), client=client)
    items = json.loads((ROOT / "experiments/data" / data_name).read_text(encoding="utf-8"))

    def one(it):
        known = dict(it.get("context") or {})
        if it.get("turns"):
            u = understand(it["turns"][0], known, client=client)
            known.update({k: v for k, v in (u.get("slots") or {}).items() if v is not None})
            known["_prev_utterance"], known["_prev_intent"] = it["turns"][0], u.get("intent")
        a = pipe.answer(it["question"], known)
        reg, missing, forbidden = regex_judge(it, a.text)
        judge, reason = model_judge(it, a.text, client)
        return {"id": it["id"], "question": it["question"], "text": a.text,
                "route": a.route, "regex": reg, "judge": judge, "reason": reason,
                "missing": missing, "forbidden": forbidden, "fallback": a.fallback_used,
                "violations": a.violations, "seconds": round(a.timings["total"], 3)}

    with ThreadPoolExecutor(max_workers=5) as ex:
        rows = list(ex.map(one, items))
    secs = sorted(r["seconds"] for r in rows)
    summary = {"n": len(rows), "regex": sum(r["regex"] for r in rows) / len(rows),
               "judge": sum(r["judge"] for r in rows) / len(rows),
               "both": sum(r["regex"] and r["judge"] for r in rows) / len(rows),
               "fallback": sum(r["fallback"] for r in rows),
               "p50_s": statistics.median(secs), "p90_s": secs[int(.9 * (len(secs) - 1))]}
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    client = cloud.llm_client()
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    data_name = sys.argv[2] if len(sys.argv) > 2 else "qa_v3.json"
    result = {}
    if mode in ("all", "first"):
        result.update(first_and_consent(client))
    if mode in ("all", "qa"):
        result["qa"] = qa(client, data_name)
    suffix = "" if data_name == "qa_v3.json" else "_" + Path(data_name).stem
    path = ROOT / "experiments/results" / f"exp10_new_approach{suffix}.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        old.update(result)
        result = old
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"first": {"correct": sum(r["expected"] == r["got"] for r in result.get("first", [])),
                                 "n": len(result.get("first", [])),
                                 "fallback": sum(r["path"] != "model" for r in result.get("first", []))},
                      "consent": {"correct": sum(r["expected"] == r["got"] for r in result.get("consent", [])),
                                  "n": len(result.get("consent", []))},
                      "qa": result.get("qa", {}).get("summary")}, ensure_ascii=False))
