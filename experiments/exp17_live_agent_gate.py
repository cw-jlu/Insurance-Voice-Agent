"""Live replay of the proposed parallel evidence gate against the current pipeline.

The same understanding and retrieval output is shared by baseline and gated arms.
Use --gate-only for the larger trigger table; --answers for paired answer quality.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from agent import cloud, small_model  # noqa: E402
from agent.pipeline import HybridRetriever, Pipeline, combine_gaps, gaps_to_check  # noqa: E402
from data_complex import items as complex_items  # noqa: E402
from embed_cache import corpus, load  # noqa: E402
from exp16_evidence_gate import ask, keyword_patterns, questions, rule  # noqa: E402
from exp2_qa_v2 import model_judge, regex_judge  # noqa: E402

OUT = ROOT / "experiments" / "results"


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    values = sorted(values)
    return {"n": len(values), "p50": round(statistics.median(values), 3),
            "p90": round(values[int(0.9 * (len(values) - 1))], 3), "max": round(values[-1], 3)}


def prepare(pipe: Pipeline, client, row: dict, catalog: str, patterns, known: dict) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        fu = pool.submit(pipe.understand, row["question"], known)
        fg = pool.submit(small_model.detect_gaps, row["question"], client)
        fr = pool.submit(pipe.retrieve, row["question"])
        fm = pool.submit(ask, client.with_options(timeout=2.0, max_retries=0), row, catalog)
        u = fu.result()
        detected = fg.result()
        retrieved = fr.result()
        model = fm.result()
    u["_retrieved"] = retrieved
    u["_gaps_check"] = gaps_to_check(u.get("gaps", []), detected)
    u["gaps"], u["_gaps_judged"] = combine_gaps(u.get("gaps", []), detected, row["question"]), True
    if u.get("intent") == "缺口" and not u["gaps"]:
        u["intent"] = "保障" if u.get("items") else "其他"
    have = set(u.get("items", [])) | {c.meta["item"] for c in retrieved if c.meta.get("item")}
    if model["predicted"] is None:
        triggered = rule(row["question"], have, patterns)
        source = "rule_fallback"
        missing = []
    else:
        missing = sorted(set(model["predicted"]) - have)
        triggered = bool(missing)
        source = "model"
    # Program-only answers and data gaps cannot benefit from the evidence agent.
    eligible = u.get("intent") not in ("报价", "优惠", "身份") and not u.get("gaps")
    gate = {"triggered": triggered, "eligible": eligible, "invoke_agent": triggered and eligible,
            "source": source, "predicted": model["predicted"], "missing": missing,
            "have": sorted(have), "model_latency_s": model["latency"],
            "parallel_prep_s": round(time.perf_counter() - t0, 3), "model_error": model["error"]}
    return u, gate


def gate_only(pipe, client, catalog, patterns, rows: list[dict], workers: int) -> dict:
    def one(row):
        u, gate = prepare(pipe, client, row, catalog, patterns, {})
        gold_missing = sorted(set(row["gold"]) - set(gate["have"]))
        return {"id": row["id"], "question": row["question"], "gold": row["gold"],
                "understood": u.get("items", []), "gold_missing": gold_missing, "gate": gate}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, rows))
    needed = [r for r in results if r["gold_missing"]]
    no_missing = [r for r in results if not r["gold_missing"]]
    summary = {"n": len(results), "missing_gold": len(needed),
               "trigger_on_missing": sum(r["gate"]["invoke_agent"] for r in needed),
               "trigger_on_complete": sum(r["gate"]["invoke_agent"] for r in no_missing),
               "agent_invocations": sum(r["gate"]["invoke_agent"] for r in results),
               "fallbacks": sum(r["gate"]["source"] == "rule_fallback" for r in results),
               "prep_s": stats([r["gate"]["parallel_prep_s"] for r in results])}
    return {"summary": summary, "rows": results}


def answer_items() -> list[dict]:
    rows = [{**r, "source": "complex"} for r in complex_items()]
    rows += [{**r, "source": "qa_v7"} for r in json.loads(
        (ROOT / "experiments" / "data" / "qa_v7.json").read_text(encoding="utf-8"))
        if r.get("valid") is not False and not r.get("turns")]
    return rows


def answers(pipe, client, catalog, patterns, rows: list[dict], workers: int) -> dict:
    def one(row):
        known = dict(row.get("context") or {})
        u, gate = prepare(pipe, client, row, catalog, patterns, known)
        out = {"id": row["id"], "source": row["source"], "question": row["question"],
               "gate": gate, "understood": u.get("items", [])}
        for name, steps in (("baseline", 0), ("gated", 5 if gate["invoke_agent"] else 0)):
            t0 = time.perf_counter()
            a = pipe.answer(row["question"], known, voice=False, agent_steps=steps,
                            agent_budget_s=2.5, understanding=copy.deepcopy(u))
            latency = time.perf_counter() - t0
            ok_r, missing, forbidden = regex_judge(row, a.text)
            try:
                ok_m, reason = model_judge(row, a.text, client)
            except Exception as exc:
                ok_m, reason = None, type(exc).__name__
            out[name] = {"text": a.text, "route": a.route, "regex": ok_r,
                         "regex_missing": missing, "regex_forbidden": forbidden,
                         "judge": ok_m, "judge_reason": reason,
                         "agent_steps": a.timings.get("agent_steps", 0),
                         "agent_s": round(a.timings.get("agent", 0), 3),
                         "answer_s": round(latency, 3),
                         "total_with_prep_s": round(gate["parallel_prep_s"] + latency, 3)}
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, row): row for row in rows}
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda r: r["id"])
    summary = {"n": len(results), "gate_triggered": sum(r["gate"]["invoke_agent"] for r in results),
               "agent_used": sum(r["gated"]["agent_steps"] > 0 for r in results),
               "fallbacks": sum(r["gate"]["source"] == "rule_fallback" for r in results)}
    for arm in ("baseline", "gated"):
        judged = [r for r in results if r[arm]["judge"] is not None]
        summary[arm] = {"regex_ok": sum(r[arm]["regex"] for r in results),
                        "judge_ok": sum(r[arm]["judge"] for r in judged),
                        "both_ok": sum(r[arm]["regex"] and r[arm]["judge"] for r in judged),
                        "judged_n": len(judged),
                        "answer_s": stats([r[arm]["answer_s"] for r in results]),
                        "total_with_prep_s": stats([r[arm]["total_with_prep_s"] for r in results]),
                        "over_10s": sum(r[arm]["total_with_prep_s"] > 10 for r in results)}
    return {"summary": summary, "rows": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--mode", choices=("gate-only", "answers"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tag", default="r1")
    args = parser.parse_args()
    if not args.env_file.is_file():
        parser.error("env file not found")
    load_dotenv(args.env_file)
    client = cloud.llm_client()
    chunks = corpus(False)
    pipe = Pipeline(HybridRetriever(chunks, load(chunks, "clauses"), client=client), client=client)
    from agent.knowledge import coverage_items
    catalog = "\n".join(f"{item['id']} {item['责任名称']}（{item['责任&加油包']}）" for item in coverage_items())
    patterns = keyword_patterns()
    rows = questions() if args.mode == "gate-only" else answer_items()
    rows = rows[args.skip:]
    if args.limit:
        rows = rows[:args.limit]
    result = (gate_only(pipe, client, catalog, patterns, rows, args.workers) if args.mode == "gate-only"
              else answers(pipe, client, catalog, patterns, rows, args.workers))
    path = OUT / f"exp17_{args.mode}_{args.tag}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
