"""实验 7：小链路"取依据"环节允许 Agent 叠多少步。

配置：固定路由（0 步）、最多 1 / 3 / 5 步（时间预算 8 秒，基本不受限），以及"最多 5 步、时间预算 3 秒"。
其余环节（理解、生成、护栏）完全相同。判分：规则 + DeepSeek（附原文）。
"""

from __future__ import annotations

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
from agent.pipeline import HybridRetriever, Pipeline  # noqa: E402
from data_complex import items  # noqa: E402
from embed_cache import corpus, load  # noqa: E402
from exp2_qa_v2 import model_judge, retry  # noqa: E402

OUT = ROOT / "experiments" / "results"
CONFIGS = [("固定路由", 0, 8.0), ("最多1步", 1, 8.0), ("最多3步", 3, 8.0), ("最多5步", 5, 8.0), ("5步/预算3秒", 5, 3.0)]


def main():
    client = cloud.llm_client()
    chunks = corpus(False)
    pipe = Pipeline(HybridRetriever(chunks, load(chunks, "clauses"), client=client), client=client)
    its = items()
    rows = []

    def one(args):
        (name, steps, budget), it = args
        t0 = time.perf_counter()
        a = retry(lambda: pipe.answer(it["question"], {}, agent_steps=steps, agent_budget_s=budget))
        lat = time.perf_counter() - t0
        rx = all(re.search(p, a.text) for p in it["must"]) and not any(re.search(p, a.text) for p in it["must_not"])
        jd, reason = retry(lambda: model_judge({**it, "turns": None}, a.text, client))
        return {"config": name, "id": it["id"], "split": it["split"], "question": it["question"], "text": a.text,
                "steps": a.timings.get("agent_steps", 0), "agent_s": round(a.timings.get("agent", 0), 3),
                "latency": round(lat, 3), "regex": rx, "judge": jd, "reason": reason, "fallback": a.fallback_used}

    jobs = [(c, it) for c in CONFIGS for it in its]
    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(one, jobs))

    summary = {}
    for name, *_ in CONFIGS:
        rs = [r for r in rows if r["config"] == name]
        lat = sorted(r["latency"] for r in rs)
        summary[name] = {
            "n": len(rs),
            "regex": round(sum(r["regex"] for r in rs) / len(rs), 4),
            "judge": round(sum(r["judge"] for r in rs) / len(rs), 4),
            "both": round(sum(r["regex"] and r["judge"] for r in rs) / len(rs), 4),
            "heldout_both": round(sum(r["regex"] and r["judge"] for r in rs if r["split"] == "heldout")
                                  / max(1, sum(r["split"] == "heldout" for r in rs)), 4),
            "steps_mean": round(statistics.mean(r["steps"] for r in rs), 2),
            "steps_max": max(r["steps"] for r in rs),
            "latency_p50": round(statistics.median(lat), 2),
            "latency_p90": round(lat[int(0.9 * (len(lat) - 1))], 2),
            "latency_max": round(lat[-1], 2),
            "fallbacks": sum(r["fallback"] for r in rs),
        }
    res = {"summary": summary, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp7_agent_steps.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main()
    for k, v in r["summary"].items():
        print(k, v)
