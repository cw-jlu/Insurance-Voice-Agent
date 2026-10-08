"""Opt-in experiment: question slots -> structured answer fields -> short spoken sentences.

The plan sees only the user question, not gold labels. Baseline and candidate reuse the
same understanding and retrieval. No text is truncated after generation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from agent import cloud  # noqa: E402
from agent.pipeline import HybridRetriever, Pipeline  # noqa: E402
from embed_cache import corpus, load  # noqa: E402
from exp18_voice_prompt_ablation import prepare, rows_for, stats  # noqa: E402
from exp2_qa_v2 import model_judge, regex_judge  # noqa: E402

PLAN_SYS = """把用户这句话概括成电话里必须回答的最少几个问题。只识别问题，不回答，不增添未问事项。
关键：用户问“一共能报多少/总价/差多少”，只列这一个总额问题，不能把每笔费用拆开。
用户问 A 和 B 的区别，只列“ A 和 B 有什么区别”，不能再拆成报销范围、比例、免赔额等未明说的问题。
同一规则适用的并列事项合成一个问题，例如“住院和门诊怎么报”合成一个；规则明显不同的事项才分开。
个人背景（年龄、社保、病史、金额）是条件，不单列。不同事项如“住院怎么报、ICU有补贴吗”分两项。
输出 JSON：{"questions":["短的子问题1","短的子问题2"]}。通常 1–2 项，最多 4 项，顺序与用户提问一致。"""


def plan(question: str, client) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    raw = cloud.chat([{"role": "system", "content": PLAN_SYS}, {"role": "user", "content": question}],
                     model=cloud.FAST_MODEL, client=client, temperature=0,
                     response_format={"type": "json_object"})
    try:
        qs = json.loads(raw).get("questions", [])
    except (ValueError, AttributeError):
        qs = []
    if not isinstance(qs, list) or not 1 <= len(qs) <= 4 or not all(isinstance(q, str) and q.strip() for q in qs):
        qs = [question]
    return qs, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--scope", choices=("complex", "all", "v5", "v6"), default="complex")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--tag", default="r1")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    client = cloud.llm_client()
    chunks = corpus(False)
    pipe = Pipeline(HybridRetriever(chunks, load(chunks, "clauses"), client=client), client=client)
    rows = rows_for(args.scope)

    def one(row):
        known = dict(row.get("context") or {})
        with ThreadPoolExecutor(2) as pool:
            fu = pool.submit(prepare, pipe, client, row)
            fp = pool.submit(plan, row["question"], client)
            u = fu.result()
            questions, plan_s = fp.result()
        results = {}
        for name in ("current", "slots"):
            this_u = copy.deepcopy(u)
            if name == "slots":
                this_u["_voice_questions"] = questions
            t0 = time.perf_counter()
            answer = pipe.answer(row["question"], known, voice=True, agent_steps=0, understanding=this_u)
            elapsed = time.perf_counter() - t0
            regex_ok, missing, forbidden = regex_judge(row, answer.text)
            judge_ok, reason = model_judge(row, answer.text, client)
            results[name] = {"text": answer.text, "chars": len(answer.text),
                             "answer_s": round(elapsed, 3), "regex_ok": regex_ok,
                             "regex_missing": missing, "regex_forbidden": forbidden,
                             "judge_ok": judge_ok, "judge_reason": reason,
                             "fallback": answer.fallback_used, "violations": answer.violations}
        return {"id": row["id"], "question": row["question"], "voice_questions": questions,
                "plan_s": round(plan_s, 3), **results}

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(one, rows))
    results.sort(key=lambda r: r["id"])
    summary = {"n": len(results), "plan_s": stats([r["plan_s"] for r in results])}
    for name in ("current", "slots"):
        summary[name] = {"both_ok": sum(r[name]["regex_ok"] and r[name]["judge_ok"] for r in results),
                         "regex_ok": sum(r[name]["regex_ok"] for r in results),
                         "judge_ok": sum(r[name]["judge_ok"] for r in results),
                         "fallback": sum(r[name]["fallback"] for r in results),
                         "chars": stats([r[name]["chars"] for r in results]),
                         "answer_s": stats([r[name]["answer_s"] for r in results])}
    output = ROOT / "experiments" / "results" / f"exp19_voice_slots_{args.scope}_{args.tag}.json"
    output.write_text(json.dumps({"summary": summary, "rows": results}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
