"""实验 15（第十轮）：方案 A–D 的消融。

方案（见 docs/08-根因分析与改进方案.md）：
  A 只选不写：模型输出"开头 + 事实编号 + 计算 + 结尾"的计划，规则句由程序从保障表生成的事实句拼接；
  B 三元组比对：小模型列出回答对各责任说的（字段, 值），程序与保障表逐格比对，不符则重写；
    B2 为第一版消融后修正：值的类型不符（如"保额=30天"）不比对，"归属"只拦"加油包说成基础版自带"；
  C 理解拆分：理解拆成意图+缺口 / 责任+主题 / 条件三个短提示词并行；
  D 数字护栏改进：允许一步算术、只在一项责任里出现的数值不再要求上下文匹配、兜底只讲被问的那一项。
配置：基线（当前版本，跑两次估计生成的随机波动）、单独加 A/B/C/D、全加、全加后各去掉一项。
题集：qa_v6（62 题，方案实现前写好，主结论用它）；qa_v4、qa_v5（119 题，已用于第九轮修改，作参考）。
判分：规则 + DeepSeek（与第九轮相同）；判分不同的题由人工复核。

用法：python experiments/exp15_ablation.py [variant ...]   # 默认全部
"""

from __future__ import annotations

import json
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
from embed_cache import corpus, load  # noqa: E402
from exp2_qa_v2 import model_judge, regex_judge, retry  # noqa: E402

OUT = ROOT / "experiments" / "results"
VARIANTS = {
    "基线": "", "基线·重复": "", "+A": "A", "+B": "B", "+C": "C", "+D": "D",
    "ABCD": "ABCD", "ABCD−A": "BCD", "ABCD−B": "ACD", "ABCD−C": "ABD", "ABCD−D": "ABC", "CD": "CD", "CD·重复": "CD", "+B2": "B", "B2CD": "BCD",
}
SETS = ["qa_v6.json", "qa_v4.json", "qa_v5.json"]


def items():
    out = []
    for s in SETS:
        for it in json.loads((ROOT / "experiments" / "data" / s).read_text(encoding="utf-8")):
            if it.get("valid", True):
                out.append({**it, "set": s.split(".")[0], "group": it.get("group", it["category"])})
    return out


def run_variant(name: str, feats: str, its: list, client, retriever) -> list[dict]:
    pipe = Pipeline(retriever, client=client, features=set(feats))

    def one(it):
        known = dict(it.get("context") or {})
        if it.get("turns"):
            u = retry(lambda: pipe.understand(it["turns"][0], known))
            known.update({k: v for k, v in (u.get("slots") or {}).items() if v is not None})
            known["_prev_utterance"], known["_prev_intent"] = it["turns"][0], u.get("intent")
        t0 = time.perf_counter()
        a = retry(lambda: pipe.answer(it["question"], known))
        lat = time.perf_counter() - t0
        rx, miss, forb = regex_judge(it, a.text)
        jd, reason = retry(lambda: model_judge(it, a.text, client))
        return {"variant": name, "id": it["id"], "set": it["set"], "group": it["group"], "category": it["category"],
                "question": it["question"], "text": a.text, "route": a.route, "fallback": a.fallback_used,
                "attempts": a.attempts, "violations": a.violations[:3], "latency": round(lat, 3),
                "understand_s": round(a.timings.get("understand", 0), 3), "regex": rx, "judge": jd, "both": rx and jd,
                "reason": reason}

    with ThreadPoolExecutor(6) as ex:
        return list(ex.map(one, its))


def summarize(rows: list[dict]) -> dict:
    def acc(rs):
        return round(sum(r["both"] for r in rs) / len(rs), 4) if rs else None

    lat = sorted(r["latency"] for r in rows)
    und = sorted(r["understand_s"] for r in rows if r["understand_s"])
    by_set = {s: acc([r for r in rows if r["set"] == s]) for s in ("qa_v6", "qa_v4", "qa_v5")}
    by_group = {g: acc([r for r in rows if r["set"] == "qa_v6" and r["group"] == g])
                for g in ("跨责任", "计算", "等待期", "路由", "常规")}
    return {"n": len(rows), "both_all": acc(rows), "by_set": by_set, "v6_by_group": by_group,
            "fallback": sum(r["fallback"] for r in rows), "rewrites": sum(r["attempts"] > 1 for r in rows),
            "latency_p50": round(statistics.median(lat), 2), "latency_p90": round(lat[int(0.9 * (len(lat) - 1))], 2),
            "understand_p50": round(statistics.median(und), 2) if und else None}


def main(names: list[str]):
    client = cloud.llm_client()
    chunks = corpus(False)
    retriever = HybridRetriever(chunks, load(chunks, "clauses"), client=client)
    its = items()
    path = OUT / "exp15_ablation.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"summary": {}, "rows": {}}
    for name in names:
        rows = run_variant(name, VARIANTS[name], its, client, retriever)
        res["rows"][name] = rows
        res["summary"][name] = summarize(rows)
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")  # save after each variant
        print(name, json.dumps(res["summary"][name], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(VARIANTS))
