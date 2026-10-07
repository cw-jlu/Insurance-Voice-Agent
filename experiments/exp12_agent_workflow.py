"""实验 12：复合问题上，固定流程与 Agent 如何结合。

四种做法，每题共用同一次理解结果（公平比较，理解耗时不计入）：
- A 固定流程：理解列出的责任 / 主题 + 一次语义检索，不调 Agent；
- B 现状：A + Agent（最多 5 步，硬截止 2.5 s，与大链路相同）；
- C 结合：A + 只在"问句提到的责任不在已取资料里"时才调 Agent（关键词检查，不调模型）；
- D 关键词补齐：A + 把关键词指向的责任直接加进资料，不调 Agent。
关键词表按保障计划表的责任名称在运行前写好，未依据本实验结果调整。
题目：data_complex 的 30 道复合题 + qa_v4 中按大链路规则（_is_complex）判为复合的题。
判分：规则 + DeepSeek（exp2_qa_v2.model_judge），之后逐题复核。
"""

from __future__ import annotations

import copy
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
from agent.pipeline import HybridRetriever, Pipeline, understand  # noqa: E402
from agent.session import _is_complex  # noqa: E402
from data_complex import items as complex_items  # noqa: E402
from embed_cache import corpus, load  # noqa: E402
from exp2_qa_v2 import model_judge, regex_judge, retry  # noqa: E402

OUT = ROOT / "experiments" / "results"
BUDGET = 2.5

# 问句关键词 → 对应责任（任一在资料里即视为已覆盖）
KEYWORDS = [
    (r"ICU|重症监护", ["R01", "R15"]),
    (r"护工|陪护", ["R08"]),
    (r"质子|重离子|硼中子|光免疫|先进疗法", ["R05"]),
    (r"靶向|特药|特定药|院外.{0,2}药|罕见病", ["R06"]),
    (r"康复", ["R04"]),
    (r"路费|交通|住宿|外省|异地|转诊|机票|火车", ["R07"]),
    (r"门诊|急诊|看病开药", ["R02", "R13"]),
    (r"津贴|补贴|每天给", ["R14", "R15"]),
    (r"网上.{0,2}药|网上药|互联网.{0,2}药|药省保|买药", ["R16"]),
    (r"一次性|直接(拿|给)钱|确诊.{0,4}(给|赔)钱|重疾关爱", ["R10"]),
    (r"特需|VIP|国际部|高端病房", ["R11"]),
    (r"没到.{0,4}(1万|一万)|不到.{0,4}(1万|一万)|免赔额.{0,4}(以内|没到)|住院.{0,2}补偿", ["R12"]),
    (r"一起算|共享免赔|全家.{0,4}免赔|家庭.{0,4}免赔", ["R09"]),
    (r"重疾|大病|癌症|恶性肿瘤|中风|尿毒症", ["R03"]),
    (r"住院", ["R01"]),
]


def keyword_items(question: str) -> list[list[str]]:
    return [ids for pat, ids in KEYWORDS if re.search(pat, question)]


def evidence_items(ans) -> set:
    return {c.chunk_id.split(":")[0].split("_")[0] for c in ans.evidence} | set(ans.understanding.get("items", []))


def build_items():
    rows = []
    for it in complex_items():
        rows.append({**it, "context": {}, "turns": None, "source": "complex"})
    for it in json.loads((ROOT / "experiments" / "data" / "qa_v4.json").read_text(encoding="utf-8")):
        if not it.get("turns"):
            rows.append({**it, "source": "v4"})
    return rows


def main():
    client = cloud.llm_client()
    ch = corpus(False)
    pipe = Pipeline(HybridRetriever(ch, load(ch, "clauses"), client=client), client=client)
    cands = build_items()

    def prep(it):
        u = retry(lambda: understand(it["question"], dict(it.get("context") or {}), client=client))
        return it, u

    with ThreadPoolExecutor(8) as ex:
        prepped = list(ex.map(prep, cands))
    # 复合题：data_complex 全部；qa_v4 中按大链路规则判为复合的题
    test = [(it, u) for it, u in prepped if it["source"] == "complex" or _is_complex(u, it["question"])]

    def run(it, u, arm):
        known = dict(it.get("context") or {})
        uu = copy.deepcopy(u)
        steps, trig = 0, None
        if arm == "B":
            steps = 5
        if arm in ("C", "D"):
            have = set(uu.get("items", []))
            miss = [ids for ids in keyword_items(it["question"]) if not (set(ids) & have)]
            trig = bool(miss)
            if arm == "C" and miss:
                steps = 5
            if arm == "D":
                for ids in miss:
                    uu["items"] = uu.get("items", []) + [ids[0]]
        t0 = time.perf_counter()
        a = retry(lambda: pipe.answer(it["question"], known, voice=False, agent_steps=steps, agent_budget_s=BUDGET,
                                      understanding=copy.deepcopy(uu)))
        lat = time.perf_counter() - t0
        ok_r, miss_r, forb = regex_judge(it, a.text)
        ok_m, reason = retry(lambda: model_judge(it, a.text, client))
        return {"arm": arm, "text": a.text, "latency": round(lat, 3), "agent_steps": a.timings.get("agent_steps", 0),
                "triggered": trig, "items_used": sorted(uu.get("items", [])),
                "evidence_ids": [c.chunk_id for c in a.evidence], "regex": ok_r, "judge": ok_m, "judge_reason": reason,
                "fallback": a.fallback_used}

    def one(pair):
        it, u = pair
        row = {"id": it["id"], "source": it["source"], "question": it["question"], "facts": it["facts"],
               "understood_items": u.get("items", []), "kw_items": keyword_items(it["question"])}
        for arm in ("A", "B", "C", "D"):
            row[arm] = run(it, u, arm)
        return row

    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(one, test))

    def summ(arm, sel=lambda r: True):
        rs = [r for r in rows if sel(r)]
        n = len(rs)
        lat = sorted(r[arm]["latency"] for r in rs)
        return {"n": n, "regex": round(sum(r[arm]["regex"] for r in rs) / n, 4),
                "judge": round(sum(r[arm]["judge"] for r in rs) / n, 4),
                "both": round(sum(r[arm]["regex"] and r[arm]["judge"] for r in rs) / n, 4),
                "latency_p50": round(statistics.median(lat), 2), "latency_p90": round(lat[int(0.9 * (n - 1))], 2),
                "agent_used": sum(1 for r in rs if r[arm]["agent_steps"]),
                "agent_steps_mean": round(statistics.mean(r[arm]["agent_steps"] for r in rs), 2)}

    res = {arm: {"all": summ(arm), "complex30": summ(arm, lambda r: r["source"] == "complex"),
                 "v4": summ(arm, lambda r: r["source"] == "v4")} for arm in ("A", "B", "C", "D")}
    # B 中 Agent 多取到的资料（相对 A）
    extra = []
    for r in rows:
        a_ids, b_ids = set(r["A"]["evidence_ids"]), set(r["B"]["evidence_ids"])
        extra.append(sorted(b_ids - a_ids))
    res["B_extra_evidence"] = {"questions_with_extra": sum(1 for e in extra if e),
                               "mean_extra_chunks": round(statistics.mean(len(e) for e in extra), 2)}
    for r, e in zip(rows, extra):
        r["B_extra_evidence"] = e
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp12_agent_workflow.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    return res


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=1))
