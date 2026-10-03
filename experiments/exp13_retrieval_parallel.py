"""实验 14a：检索能否不等理解模块、直接用用户原话与理解并行。

现在的顺序：理解（约 1 s，输出 rewrite）→ 用 rewrite 检索（向量化约 0.1–0.3 s）。
如果用原话检索的命中率不低于用 rewrite，就可以把检索提前到与理解并行，省掉检索这一段串行时间。
题集：第三轮条款检索 143 题（exp_retrieval_v3 的有效题），金标准与之前相同。
比较：原话、理解模块的 rewrite（当前生产做法）、原话 + rewrite 拼接。

用法：python experiments/exp13_retrieval_parallel.py
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agent import cloud  # noqa: E402
from agent.pipeline import HybridRetriever, understand  # noqa: E402
from embed_cache import corpus, load  # noqa: E402
import exp_retrieval as v1  # noqa: E402
import exp_retrieval_v3 as v3  # noqa: E402

OUT = ROOT / "experiments" / "results"


def main():
    chunks = corpus(False)
    queries = v1.QUERIES + [(q, c) for q, c in v3.NEW if any(v3.hit(ch, c) for ch in chunks)]
    client = cloud.llm_client()
    r = HybridRetriever(chunks, load(chunks, "clauses"), client=client)
    with ThreadPoolExecutor(8) as ex:
        rewrites = list(ex.map(lambda q: (understand(q[0], {}, client=client).get("rewrite") or q[0]), queries))

    def rank(query, conds):
        res = r.search(query, k=10)
        return next((i + 1 for i, c in enumerate(res) if v3.hit(c, conds)), None)

    out = {}
    for name, qf in (("原话", lambda i: queries[i][0]), ("理解模块rewrite（当前）", lambda i: rewrites[i]),
                     ("原话+rewrite", lambda i: f"{queries[i][0]} {rewrites[i]}")):
        with ThreadPoolExecutor(8) as ex:
            ranks = list(ex.map(lambda i: rank(qf(i), queries[i][1]), range(len(queries))))
        n = len(queries)
        out[name] = {"hit@1": round(sum(1 for x in ranks if x and x <= 1) / n, 4), "hit@3": round(sum(1 for x in ranks if x and x <= 3) / n, 4),
                     "hit@5": round(sum(1 for x in ranks if x and x <= 5) / n, 4), "ranks": ranks}
    res = {"n": len(queries), "methods": {k: {a: b for a, b in v.items() if a != "ranks"} for k, v in out.items()},
           "only_raw_hit5": [queries[i][0] for i in range(len(queries)) if (out["原话"]["ranks"][i] or 99) <= 5 < (out["理解模块rewrite（当前）"]["ranks"][i] or 99)],
           "only_rewrite_hit5": [queries[i][0] for i in range(len(queries)) if (out["理解模块rewrite（当前）"]["ranks"][i] or 99) <= 5 < (out["原话"]["ranks"][i] or 99)],
           "rewrites": dict(zip([q for q, _ in queries], rewrites))}
    (OUT / "exp13_retrieval_parallel.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(res["n"], json.dumps(res["methods"], ensure_ascii=False))
    print("only raw:", res["only_raw_hit5"]); print("only rewrite:", res["only_rewrite_hit5"])


if __name__ == "__main__":
    main()
