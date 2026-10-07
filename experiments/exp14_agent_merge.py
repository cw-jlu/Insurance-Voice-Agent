"""实验 14b：复合问题的"Agent 叠 step"能否并入理解层（一次选全依据），效果不变、耗时更低。

Agent 每一步是一次模型调用，最后还要再调用一次判断"够了"，所以叠 1 步约多 2.5–3 s（第六轮重测）。
理解模块本来就会选相关责任（最多 3 项）；如果放宽到最多 5 项、让它"问到几项就列几项"，
复合问题的依据也许一次就能取齐，不需要 Agent。
题集：第五轮 30 道复合问题（data_complex，开发 20 / 保留 10），判分与第五轮相同（规则 + DeepSeek，附原文）。
配置：A 理解≤3 项 + Agent≤5 步（当前做法）；B 理解≤5 项、不用 Agent；C 理解≤5 项 + Agent≤5 步。

用法：python experiments/exp14_agent_merge.py
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

from agent import cloud, pipeline  # noqa: E402
from agent.pipeline import HybridRetriever, Pipeline  # noqa: E402
from data_complex import items  # noqa: E402
from embed_cache import corpus, load  # noqa: E402
from exp2_qa_v2 import model_judge, retry  # noqa: E402

OUT = ROOT / "experiments" / "results"
CONFIGS = [("A 理解≤3+Agent≤5步", 3, 5), ("B 理解≤5、无Agent", 5, 0), ("C 理解≤5+Agent≤5步", 5, 5)]


def main():
    client = cloud.llm_client()
    chunks = corpus(False)
    pipe = Pipeline(HybridRetriever(chunks, load(chunks, "clauses"), client=client), client=client)
    its = items()

    def one(args):
        (name, max_items, steps), it = args
        from agent.small_model import detect_gaps
        from agent.pipeline import combine_gaps, gaps_to_check, understand
        t0 = time.perf_counter()

        def ans():
            u = understand(it["question"], {}, client=client, max_items=max_items)
            d = detect_gaps(it["question"], client)
            u["_gaps_check"], u["gaps"], u["_gaps_judged"] = gaps_to_check(u.get("gaps", []), d), combine_gaps(u.get("gaps", []), d), True
            if u.get("intent") == "缺口" and not u["gaps"]:
                u["intent"] = "保障" if u.get("items") else "其他"
            return pipe.answer(it["question"], {}, agent_steps=steps, agent_budget_s=8.0, understanding=u)

        a = retry(ans)
        lat = time.perf_counter() - t0
        rx = all(re.search(p, a.text) for p in it["must"]) and not any(re.search(p, a.text) for p in it["must_not"])
        jd, reason = retry(lambda: model_judge({**it, "turns": None}, a.text, client))
        return {"config": name, "id": it["id"], "split": it["split"], "question": it["question"], "text": a.text,
                "items": a.understanding.get("items"), "steps": a.timings.get("agent_steps", 0), "latency": round(lat, 3),
                "regex": rx, "judge": jd, "reason": reason, "fallback": a.fallback_used}

    jobs = [(c, it) for c in CONFIGS for it in its]
    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(one, jobs))
    summary = {}
    for name, *_ in CONFIGS:
        rs = [r for r in rows if r["config"] == name]
        lat = sorted(r["latency"] for r in rs)
        summary[name] = {"n": len(rs), "regex": round(sum(r["regex"] for r in rs) / len(rs), 4),
                         "judge": round(sum(r["judge"] for r in rs) / len(rs), 4),
                         "both": round(sum(r["regex"] and r["judge"] for r in rs) / len(rs), 4),
                         "heldout_both": f"{sum(r['regex'] and r['judge'] for r in rs if r['split'] == 'heldout')}/{sum(r['split'] == 'heldout' for r in rs)}",
                         "items_mean": round(statistics.mean(len(r["items"] or []) for r in rs), 2),
                         "steps_mean": round(statistics.mean(r["steps"] for r in rs), 2),
                         "latency_p50": round(statistics.median(lat), 2), "latency_p90": round(lat[int(0.9 * (len(lat) - 1))], 2),
                         "latency_max": round(lat[-1], 2), "fallbacks": sum(r["fallback"] for r in rs)}
    (OUT / "exp14_agent_merge.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in summary.items():
        print(k, json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    main()
