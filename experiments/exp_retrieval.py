"""实验 2（检索部分）：条款检索召回率。

语料：保障计划表每个单元格 + 投保须知 + 责任免除（按行切片，带出处）。
查询：用客户口语写的问题，人工标注应命中的责任项或条款关键词。
对比：BM25（字二元组）、TF-IDF（字 1–3 元组）、两者 RRF 融合、融合 + 口语改写。
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.guardrail import ALIASES  # noqa: E402
from agent.knowledge import Chunk, clause_chunks  # noqa: E402
from agent.retrieval import Retriever  # noqa: E402

OUT = ROOT / "experiments" / "results"

# (查询, 命中条件)。命中条件：("item", 责任编号) 表示命中该责任的任一单元格；
# ("doc", 文档, 关键词) 表示命中该文档中包含关键词的切片。满足任一条件即算命中。
QUERIES: list[tuple[str, list[tuple]]] = [
    ("一般住院自己要先出多少钱", [("item", "R01")]),
    ("门诊看病能报吗", [("item", "R02"), ("item", "R13")]),
    ("得了重病住院有没有免赔额", [("item", "R03")]),
    ("中风以后去康复医院住院能报多少", [("item", "R04")]),
    ("质子重离子治疗能报吗", [("item", "R05")]),
    ("癌症的靶向药能不能报", [("item", "R06")]),
    ("基因检测费用报不报", [("item", "R06")]),
    ("去外省看大病的路费住宿能报吗", [("item", "R07")]),
    ("住院请护工的钱给不给", [("item", "R08")]),
    ("一家人能不能一起算一个免赔额", [("item", "R09")]),
    ("确诊重疾能一次性给钱吗", [("item", "R10")]),
    ("住特需病房VIP病房能报吗", [("item", "R11")]),
    ("一万以下的住院费有没有办法报", [("item", "R12")]),
    ("看门诊每次能报多少，有上限吗", [("item", "R13")]),
    ("住院每天能补贴多少钱", [("item", "R14")]),
    ("进ICU有没有额外补贴", [("item", "R15")]),
    ("网上买药能报吗", [("item", "R16")]),
    ("以前就有的病能报吗", [("doc", "责任免除", "既往症")]),
    ("医保已经报了的部分还能再报吗", [("doc", "责任免除", "补偿原则")]),
    ("私立医院看病能报吗", [("doc", "责任免除", "非中国公立二级"), ("item", "R01")]),
    ("刚买完就生病住院能赔吗", [("doc", "责任免除", "等待期内"), ("doc", "投保须知", "等待期")]),
    ("做美容整形能报吗", [("doc", "责任免除", "整形")]),
    ("生孩子的费用报不报", [("doc", "责任免除", "分娩")]),
    ("看牙能报吗", [("doc", "责任免除", "牙科")]),
    ("得了抑郁症住院能报吗", [("doc", "责任免除", "精神和行为障碍")]),
    ("工作中得了职业病能报吗", [("doc", "责任免除", "职业病")]),
    ("多大年纪可以买", [("doc", "投保须知", "被保险人年龄")]),
    ("我能给我爸妈买吗", [("doc", "投保须知", "投保人资格"), ("doc", "投保须知", "被保险人")]),
    ("我是开货车的能买吗", [("doc", "投保须知", "职业")]),
    ("能不能多买几份", [("doc", "投保须知", "限投保1份")]),
    ("买了什么时候开始生效", [("doc", "投保须知", "次日零时")]),
    ("等待期是多久", [("doc", "投保须知", "等待期"), ("item", "R01")]),
    ("保险是一年的还是长期的", [("doc", "投保须知", "保险期间")]),
    ("价格每年会变吗", [("doc", "投保须知", "随着您的年龄增长")]),
    ("不如实告知会怎样", [("doc", "投保须知", "如实告知")]),
    ("没用医保结算报多少", [("item", "R01"), ("item", "R03")]),
    ("门急诊加油包多大年纪能买", [("doc", "投保须知", "门急诊医疗加油包"), ("item", "R13")]),
    ("特药要在哪里买", [("item", "R06")]),
    ("海南博鳌的进口药能报吗", [("item", "R06")]),
    ("罕见病用药能报吗", [("item", "R06")]),
]


def hit(c: Chunk, conds: list[tuple]) -> bool:
    for cond in conds:
        if cond[0] == "item" and c.meta.get("item") == cond[1]:
            return True
        if cond[0] == "doc" and c.doc == cond[1] and cond[2] in c.text:
            return True
    return False


def rewrite(q: str) -> str:
    return q + " " + " ".join(v for k, v in ALIASES.items() if k in q)


def evaluate(r: Retriever, method: str, use_rewrite: bool) -> dict:
    ranks, lat, misses = [], [], []
    for q, conds in QUERIES:
        query = rewrite(q) if use_rewrite else q
        t0 = time.perf_counter()
        res = r.search(query, k=10, method=method)
        lat.append((time.perf_counter() - t0) * 1000)
        rank = next((i + 1 for i, c in enumerate(res) if hit(c, conds)), None)
        ranks.append(rank)
        if rank is None or rank > 5:
            misses.append({"query": q, "top1": f"{res[0].source} {res[0].text[:40]}"})
    n = len(QUERIES)
    return {
        "hit@1": round(sum(1 for x in ranks if x and x <= 1) / n, 4),
        "hit@3": round(sum(1 for x in ranks if x and x <= 3) / n, 4),
        "hit@5": round(sum(1 for x in ranks if x and x <= 5) / n, 4),
        "mrr@10": round(sum(1 / x for x in ranks if x) / n, 4),
        "latency_ms_p50": round(statistics.median(lat), 2),
        "latency_ms_max": round(max(lat), 2),
        "misses@5": misses,
    }


def main() -> dict:
    chunks = clause_chunks()
    t0 = time.perf_counter()
    r = Retriever(chunks)
    build_ms = (time.perf_counter() - t0) * 1000
    result = {
        "corpus_chunks": len(chunks),
        "queries": len(QUERIES),
        "index_build_ms": round(build_ms, 1),
        "methods": {
            "bm25": evaluate(r, "bm25", False),
            "tfidf": evaluate(r, "tfidf", False),
            "hybrid": evaluate(r, "hybrid", False),
            "hybrid+rewrite": evaluate(r, "hybrid", True),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp_retrieval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    res = main()
    print("chunks", res["corpus_chunks"], "queries", res["queries"], "build_ms", res["index_build_ms"])
    for m, v in res["methods"].items():
        print(m, {k: x for k, x in v.items() if k != "misses@5"})
    print("hybrid+rewrite misses:", json.dumps(res["methods"]["hybrid+rewrite"]["misses@5"], ensure_ascii=False, indent=1))
