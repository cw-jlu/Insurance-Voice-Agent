"""实验 2（检索部分）第二轮：加入语义向量和 LLM 查询改写，沿用第一轮的 40 个问题和标注。"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agent import cloud  # noqa: E402
from agent.pipeline import HybridRetriever  # noqa: E402
from embed_cache import corpus, load  # noqa: E402
from exp_retrieval import QUERIES, hit  # noqa: E402

OUT = ROOT / "experiments" / "results"

REWRITE_SYS = ("把用户关于百万医疗保险的口语问题改写成保险条款里的用语，用于检索条款。只输出改写后的一句话，不要解释。"
               "例如：'以前就有的病能报吗' → '既往症 是否理赔'；'买了什么时候管用' → '保单生效日期'。")


def rewrite_all(client) -> tuple[dict[str, str], list[float]]:
    out, lat = {}, []
    for q, _ in QUERIES:
        t0 = time.perf_counter()
        out[q] = cloud.chat([{"role": "system", "content": REWRITE_SYS}, {"role": "user", "content": q}],
                            model=cloud.FAST_MODEL, client=client, temperature=0).strip()
        lat.append(time.perf_counter() - t0)
    return out, lat


def evaluate(r: HybridRetriever, mode: str, rewrites: dict | None) -> dict:
    ranks, lat, misses = [], [], []
    for q, conds in QUERIES:
        query = f"{q} {rewrites[q]}" if rewrites else q
        t0 = time.perf_counter()
        res = r.search(query, k=10, mode=mode)
        lat.append((time.perf_counter() - t0) * 1000)
        rank = next((i + 1 for i, c in enumerate(res) if hit(c, conds)), None)
        ranks.append(rank)
        if rank is None or rank > 5:
            misses.append({"query": q, "rewrite": rewrites.get(q) if rewrites else None, "top1": f"{res[0].source} {res[0].text[:40]}"})
    n = len(QUERIES)
    return {
        "hit@1": round(sum(1 for x in ranks if x and x <= 1) / n, 4),
        "hit@3": round(sum(1 for x in ranks if x and x <= 3) / n, 4),
        "hit@5": round(sum(1 for x in ranks if x and x <= 5) / n, 4),
        "mrr@10": round(sum(1 / x for x in ranks if x) / n, 4),
        "latency_ms_p50": round(statistics.median(lat), 1),
        "misses@5": misses,
    }


def main() -> dict:
    client = cloud.llm_client()
    chunks = corpus(False)
    r = HybridRetriever(chunks, load(chunks, "clauses"), client=client)
    rewrites, rw_lat = rewrite_all(client)
    result = {
        "embed_model": cloud.EMBED_MODEL,
        "rewrite_model": cloud.FAST_MODEL,
        "rewrite_latency_s_p50": round(statistics.median(rw_lat), 3),
        "methods": {
            "tfidf (第一轮最佳)": evaluate(r, "lexical", None),
            "dense": evaluate(r, "dense", None),
            "tfidf+dense": evaluate(r, "hybrid", None),
            "dense+改写": evaluate(r, "dense", rewrites),
            "tfidf+dense+改写": evaluate(r, "hybrid", rewrites),
        },
        "rewrites": rewrites,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp_retrieval_v2.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    res = main()
    print("rewrite p50", res["rewrite_latency_s_p50"])
    for m, v in res["methods"].items():
        print(m, {k: x for k, x in v.items() if k != "misses@5"})
    print(json.dumps(res["methods"]["tfidf+dense+改写"]["misses@5"], ensure_ascii=False, indent=1))
