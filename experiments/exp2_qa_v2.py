"""实验 2 扩充：端到端问答，使用冻结的 qa_v2.json（质检后有效题目）。

判分两套：
- 规则判分：must / must_not 正则；
- 模型判分：DeepSeek（与生成用的 qwen 不同厂商），对照"标准要点"判断回答是否正确、有无资料外说法。
最终在保留集上人工抽检，估计两套判分与人工的一致率。
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
from agent.pipeline import HybridRetriever, Pipeline, baseline_answer, understand  # noqa: E402
from embed_cache import corpus, load  # noqa: E402

OUT = ROOT / "experiments" / "results"
DATA = ROOT / "experiments" / "data" / "qa_v2.json"
AGENT_STEPS = 0
JUDGE_MODEL = "deepseek-v4-flash"
JUDGE_SYS = """你是保险电销质检员。给你：用户问题、标准要点（来自保险资料，是判分依据）、AI 回答。
判断 AI 回答是否合格：
- 合格：回答的核心结论与标准要点一致；没有说错数字或规则；没有资料以外的承诺或结论。可以省略次要信息，可以多说与问题相关且正确的内容。
- 标准要点只列核心，不是全部资料；如果附有【原文】，以原文为准判断细节。回答中额外的细节只要与原文一致就不算错；原文没有提到、无法确定对错的细节也不算错。
- 回答里出现"我是……AI助理"即表示已承认是 AI。
- 不合格：核心结论错误或缺失；说错数字、比例、年龄、天数；把一项责任的规则说成另一项；对"需以核保/条款为准"的问题给出了肯定结论；该追问条件时没有追问而直接报了价。
只输出 JSON：{"correct": true/false, "reason": "一句话原因"}"""


def regex_judge(it, text):
    missing = [p for p in it["must"] if not re.search(p, text)]
    forbidden = [p for p in it["must_not"] if re.search(p, text)]
    return not missing and not forbidden, missing, forbidden


def source_text(facts: str) -> str:
    """Full coverage-table text for the items a reference answer cites (R01–R16)."""
    from agent.knowledge import coverage_items

    ids = set(re.findall(r"R\d\d", facts))
    return "\n".join(f"{it['责任名称']}（{it['责任&加油包']}）：" + "；".join(c["text"] for c in it["责任描述"])
                     for it in coverage_items() if it["id"] in ids)


def model_judge(it, text, client):
    q = " → ".join(it["turns"]) if it.get("turns") else it["question"]
    src = source_text(it["facts"])
    content = f"用户问题：{q}\n标准要点：{it['facts']}\n" + (f"【原文】\n{src}\n" if src else "") + f"AI 回答：{text}"
    raw = cloud.chat([{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": content}],
                     model=JUDGE_MODEL, client=client, temperature=0, response_format={"type": "json_object"})
    try:
        d = json.loads(raw)
        return bool(d.get("correct")), d.get("reason", "")
    except json.JSONDecodeError:
        return False, "judge parse error"


def retry(fn, n=4):
    for i in range(n):
        try:
            return fn()
        except Exception:
            if i == n - 1:
                raise
            time.sleep(2 + 3 * i)


FACT_CHECK = "off"
BASELINE = True


def run(tag: str, workers: int = 6):
    client = cloud.llm_client()
    clauses = corpus(False)
    pipe = Pipeline(HybridRetriever(clauses, load(clauses, "clauses"), client=client), client=client, fact_check=FACT_CHECK)
    with_rates = corpus(True)
    base_ret = HybridRetriever(with_rates, load(with_rates, "with_rates"), client=client)
    items = [i for i in json.loads(DATA.read_text(encoding="utf-8")) if i["valid"]]

    def one(it):
        row = {k: it[k] for k in ("id", "split", "category", "source", "question")}
        known = dict(it.get("context") or {})
        if it.get("turns"):
            u = retry(lambda: understand(it["turns"][0], known, client=client))
            known.update({k: v for k, v in (u.get("slots") or {}).items() if v is not None})
            known["_prev_utterance"], known["_prev_intent"] = it["turns"][0], u.get("intent")
        q = it["question"]
        def ours():
            if not AGENT_STEPS:
                return pipe.answer(q, known)
            from agent.session import _is_complex
            u = understand(q, known, client=client)
            steps = AGENT_STEPS if _is_complex(u, q) else 0
            return pipe.answer(q, known, agent_steps=steps, agent_budget_s=4.0, understanding=u)

        variants = [("pipeline", ours)]
        if BASELINE:
            variants.append(("baseline", lambda: baseline_answer(" ".join(it["turns"]) if it.get("turns") else q, base_ret, client=client)))
        for name, fn in variants:
            a = retry(fn)
            ok_r, miss, forb = regex_judge(it, a.text)
            ok_m, reason = retry(lambda: model_judge(it, a.text, client))
            row[name] = {"text": a.text, "route": a.route, "fallback": a.fallback_used, "violations": a.violations,
                         "latency": round(a.timings.get("total", 0), 3), "regex": ok_r, "missing": miss, "forbidden": forb,
                         "judge": ok_m, "judge_reason": reason}
        return row

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, items))

    def summ(sel, name):
        rs = [r for r in rows if sel(r)]
        if not rs:
            return None
        n = len(rs)
        lat = sorted(r[name]["latency"] for r in rs)
        return {"n": n, "regex": round(sum(r[name]["regex"] for r in rs) / n, 4),
                "judge": round(sum(r[name]["judge"] for r in rs) / n, 4),
                "both": round(sum(r[name]["regex"] and r[name]["judge"] for r in rs) / n, 4),
                "fallback": sum(r[name]["fallback"] for r in rs),
                "latency_p50": round(statistics.median(lat), 2), "latency_p90": round(lat[int(0.9 * (n - 1))], 2)}

    cats = sorted({r["category"] for r in rows})
    res = {"tag": tag, "judge_model": JUDGE_MODEL, "items": len(rows)}
    for name in ("pipeline", "baseline") if BASELINE else ("pipeline",):
        res[name] = {"dev": summ(lambda r: r["split"] == "dev", name),
                     "heldout": summ(lambda r: r["split"] == "heldout", name),
                     "heldout_paraphrase": summ(lambda r: r["split"] == "heldout" and r["source"] == "改写", name),
                     "by_category_heldout_judge": {c: (summ(lambda r, c=c: r["split"] == "heldout" and r["category"] == c, name) or {}).get("judge") for c in cats}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"exp2_qa_v2_{tag}.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return res, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--show-dev-errors", action="store_true")
    ap.add_argument("--data", default="qa_v2.json")
    ap.add_argument("--agent-steps", type=int, default=0)
    ap.add_argument("--fact-check", default="off", choices=["off", "multi", "always"])
    ap.add_argument("--no-baseline", action="store_true")
    a = ap.parse_args()
    FACT_CHECK, BASELINE = a.fact_check, not a.no_baseline
    DATA = ROOT / "experiments" / "data" / a.data
    AGENT_STEPS = a.agent_steps
    res, rows = run(a.tag)
    for name in ("pipeline", "baseline") if BASELINE else ("pipeline",):
        for split in ("dev", "heldout", "heldout_paraphrase"):
            print(name, split, res[name][split])
    print("by category (heldout, judge):", res["pipeline"]["by_category_heldout_judge"])
    if a.show_dev_errors:
        for r in rows:
            p = r["pipeline"]
            if r["split"] == "dev" and not (p["regex"] and p["judge"]):
                print(f"\n[{r['id']} {r['category']}] {r['question']}\n  regex={p['regex']} judge={p['judge']} ({p['judge_reason']}) route={p['route']} fb={p['fallback']}"
                      f"\n  missing={p['missing']} forbidden={p['forbidden']} viol={p['violations'][:2]}\n  答：{p['text']}")
