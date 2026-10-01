"""实验 2：端到端问答准确率。本方案流水线 vs 纯 RAG 基线。

判定：must 中每个要点都命中、must_not 无一命中，记为正确。规则判定对"说法不同但正确"的回答可能误判，
错误样例在报告中人工复核。调整提示词时只看开发集的错误；保留集只输出汇总数字。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agent import cloud  # noqa: E402
from agent.guardrail import compliance_guard, gap_guard  # noqa: E402
from agent.pipeline import HybridRetriever, Pipeline, baseline_answer  # noqa: E402
from embed_cache import corpus, load  # noqa: E402
from qa_set import items  # noqa: E402

OUT = ROOT / "experiments" / "results"
VOICE = False


def judge(item: dict, text: str) -> dict:
    missing = [p for p in item["must"] if not re.search(p, text)]
    forbidden = [p for p in item["must_not"] if re.search(p, text)]
    return {"correct": not missing and not forbidden, "missing": missing, "forbidden": forbidden}


def run(tag: str, workers: int = 4) -> dict:
    client = cloud.llm_client()
    clauses = corpus(False)
    pipe = Pipeline(HybridRetriever(clauses, load(clauses, "clauses"), client=client), client=client)
    with_rates = corpus(True)
    base_ret = HybridRetriever(with_rates, load(with_rates, "with_rates"), client=client)
    its = items()

    def one(it):
        out = {"id": it["id"], "split": it["split"], "category": it["category"], "question": it["question"]}
        for name, fn in (("pipeline", lambda: pipe.answer(it["question"], it["context"], voice=VOICE)),
                         ("baseline", lambda: baseline_answer(it["question"], base_ret, client=client))):
            for attempt in range(3):
                try:
                    a = fn()
                    break
                except Exception as e:  # transient API errors
                    if attempt == 2:
                        raise
                    time.sleep(2 + attempt * 3)
            j = judge(it, a.text)
            out[name] = {"text": a.text, "route": a.route, "fallback": a.fallback_used, "attempts": a.attempts,
                         "violations": a.violations, "timings": {k: round(v, 3) for k, v in a.timings.items()},
                         "gap_violation": not gap_guard(it["question"], a.text).ok,
                         "compliance_violation": not compliance_guard(a.text).ok, **j}
        return out

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, its))

    def summary(sel, name):
        rs = [r for r in rows if sel(r)]
        n = len(rs)
        ok = sum(r[name]["correct"] for r in rs)
        tot = [r[name]["timings"]["total"] for r in rs]
        return {"n": n, "correct": ok, "accuracy": round(ok / n, 4) if n else None,
                "gap_conclusions": sum(r[name]["gap_violation"] for r in rs),
                "compliance_violations": sum(r[name]["compliance_violation"] for r in rs),
                "fallback_used": sum(r[name]["fallback"] for r in rs),
                "latency_s_p50": round(statistics.median(tot), 2) if tot else None,
                "latency_s_p90": round(sorted(tot)[int(0.9 * (n - 1))], 2) if tot else None}

    cats = sorted({r["category"] for r in rows})
    res = {"tag": tag, "models": {"fast": cloud.FAST_MODEL, "slow": cloud.SLOW_MODEL, "embed": cloud.EMBED_MODEL}}
    for name in ("pipeline", "baseline"):
        res[name] = {
            "dev": summary(lambda r: r["split"] == "dev", name),
            "heldout": summary(lambda r: r["split"] == "heldout", name),
            "all": summary(lambda r: True, name),
            "by_category_all": {c: summary(lambda r, c=c: r["category"] == c, name)["accuracy"] for c in cats},
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"exp2_qa_{tag}.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary": res, "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--show-dev-errors", action="store_true")
    ap.add_argument("--voice", action="store_true", help="电话模式：短回答 + 长度上限")
    args = ap.parse_args()
    VOICE = args.voice
    out = run(args.tag)
    s = out["summary"]
    for name in ("pipeline", "baseline"):
        print(name, "dev", s[name]["dev"])
        print(name, "heldout", s[name]["heldout"])
    if args.show_dev_errors:
        for r in out["rows"]:
            if r["split"] == "dev" and not r["pipeline"]["correct"]:
                p = r["pipeline"]
                print(f"\n[{r['id']} {r['category']}] {r['question']}\n  route={p['route']} fb={p['fallback']} viol={p['violations']}"
                      f"\n  missing={p['missing']} forbidden={p['forbidden']}\n  答：{p['text']}")
