"""实验 8 补充：用 DeepSeek 逐轮审阅 24 通通话记录（不重新跑通话）。

每一轮（用户一句 + 坐席回复）判断四项：
- 回应了用户的问题吗（用户没提问时为"不适用"）
- 有事实错误吗（对照附上的产品要点）
- 有没有承诺做不到的事（系统能力只有：报价、讲保障、发投保链接短信、转人工）
- 有没有对资料中没有的事项下结论（疾病能否投保、续保、医保报销部分能否计入免赔额）
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import cloud  # noqa: E402

OUT = ROOT / "experiments" / "results"
FACTS = ("一般医疗与门急诊共享1万元年免赔额、300万，扣除后100%；以有社保身份投保但未用社保结算60%，按无社保投保不受此限；"
         "重大疾病医疗300万无免赔额；恶性肿瘤先进疗法、特定药品各600万；必选计划等待期30天，药省保7天，意外无等待期；"
         "二级及以上公立医院或指定民营医疗机构普通部；既往症不赔；被保人30天-70周岁，门急诊加油包至55周岁；"
         "保险期间1年，次日零时生效；家庭单2/3/4/5人以上优惠5%/10%/15%/20%；加油包需另外选购；"
         "资料中没有：疾病核保规则、续保规则、医保报销部分能否计入免赔额。")
SYS = f"""你在审阅保险电销 AI 坐席的一轮对话。产品要点：{FACTS}
坐席系统的能力只有：报价、讲保障、发投保链接短信、转人工。
对这一轮输出 JSON：
{{"user_asked": 用户这句话里有没有提问(true/false),
  "answered": 坐席是否回应了用户的提问(true/false，用户没提问时为 true),
  "factual_error": 坐席是否说错了事实(true/false),
  "false_promise": 坐席是否承诺了系统做不到的事，如发微信、寄资料、给额外优惠(true/false),
  "gap_conclusion": 坐席是否对资料中没有的事项给出了结论(true/false),
  "note": "有问题时一句话说明"}}"""


def review(user, agent, client):
    raw = cloud.chat([{"role": "system", "content": SYS}, {"role": "user", "content": f"用户：{user}\n坐席：{agent}"}],
                     model="deepseek-v4-flash", client=client, temperature=0, response_format={"type": "json_object"})
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def main():
    d = json.loads((OUT / "exp8_calls.json").read_text(encoding="utf-8"))
    client = cloud.llm_client()
    pairs = []
    for ci, c in enumerate(d["calls"]):
        tr = c["transcript"]
        for i, (sp, t, _) in enumerate(tr):
            if sp == "用户" and i + 1 < len(tr):
                pairs.append((ci, c["persona"], t, tr[i + 1][1]))
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda p: review(p[2], p[3], client), pairs))
    rows = [{"call": ci, "persona": pe, "user": u, "agent": a, **r} for (ci, pe, u, a), r in zip(pairs, res)]
    asked = [r for r in rows if r.get("user_asked")]
    summary = {
        "turns": len(rows),
        "turns_with_question": len(asked),
        "question_answered": f"{sum(bool(r.get('answered')) for r in asked)}/{len(asked)}",
        "factual_error_turns": sum(bool(r.get("factual_error")) for r in rows),
        "false_promise_turns": sum(bool(r.get("false_promise")) for r in rows),
        "gap_conclusion_turns": sum(bool(r.get("gap_conclusion")) for r in rows),
        "calls_with_problem": len({r["call"] for r in rows if (r.get("user_asked") and not r.get("answered"))
                                   or r.get("factual_error") or r.get("false_promise") or r.get("gap_conclusion")}),
    }
    (OUT / "exp8_turn_review.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for r in rows:
        bad = [k for k in ("factual_error", "false_promise", "gap_conclusion") if r.get(k)]
        if r.get("user_asked") and not r.get("answered"):
            bad.append("未回应提问")
        if bad:
            print(f"✗ [{r['persona']}] {bad} {r.get('note', '')}\n   用户：{r['user'][:50]}\n   坐席：{r['agent'][:70]}")


if __name__ == "__main__":
    main()
