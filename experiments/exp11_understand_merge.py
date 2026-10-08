"""实验 11：理解阶段三次调用合并为一次。

分开（现状，三路并行）：understand + small_model.classify_act + small_model.detect_gaps；
合并：pipeline.understand_merged（同一提示词里多输出 act 与 gaps_now）。模型均为 qwen-flash。

有标注的集合直接比准确率：
- 缺口检测 275 题（qa_v2 / qa_v3 单轮题，类别为"缺口"即正例）：detect_gaps vs gaps_now；
- 条件抽取 200 句：understand 的 slots vs 合并版 slots（年龄、社保）；
- 身份识别 176 例：大链路的判定规则（规则命中，或 act 与 intent 都判为身份）分别用两种理解结果。
没有标注的比一致率，不一致的另行复核：
- 整通电话中的用户语句（exp8_calls_v2 / v3 的 120 通），带最近对话：act、intent、items、gaps_now。
时延：从通话语句中抽 100 句，逐句顺序运行（不并发），分开取三路最慢、合并取单次。
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agent import cloud, small_model  # noqa: E402
from agent.guardrail import is_identity_question  # noqa: E402
from agent.pipeline import understand, understand_merged  # noqa: E402
import exp9_models as e9  # noqa: E402

OUT = ROOT / "experiments" / "results"
C = cloud.llm_client()


def retry(fn, *a):
    for i in range(4):
        try:
            return fn(*a)
        except Exception:
            if i == 3:
                raise
            time.sleep(2 + 2 * i)


def separate(text: str, hist: str = "", known: dict | None = None):
    known = {**(known or {}), **({"_history": hist} if hist else {})}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(3) as ex:
        fu = ex.submit(retry, understand, text, known, C)
        fa = ex.submit(retry, small_model.classify_act, text, hist, C)
        fg = ex.submit(retry, small_model.detect_gaps, text, C)
        u, act, gaps = fu.result(), fa.result(), fg.result()
    return {**u, "act": act, "gaps_now": gaps}, time.perf_counter() - t0


def merged(text: str, hist: str = "", known: dict | None = None):
    known = {**(known or {}), **({"_history": hist} if hist else {})}
    t0 = time.perf_counter()
    u = retry(understand_merged, text, known, C)
    return u, time.perf_counter() - t0


def identity_decision(text, u):
    return is_identity_question(text) or (u["act"] == "身份" and u.get("intent") == "身份")


def call_turns():
    rows = []
    for tag in ("v2", "v3"):
        d = json.loads((OUT / f"exp8_calls_{tag}.json").read_text(encoding="utf-8"))
        for ci, c in enumerate(d["calls"]):
            tr = c["transcript"]
            for i, t in enumerate(tr):
                if t[0] != "用户":
                    continue
                hist = "\n".join(f"{s}：{x}" for s, x, *_ in tr[max(0, i - 5): i + 1])
                rows.append({"call": f"{tag}#{ci}", "turn": i, "text": t[1], "hist": hist})
    return rows


def pmap(fn, items, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def lat(xs):
    xs = sorted(xs)
    return {"p50": round(statistics.median(xs), 3), "p90": round(xs[int(0.9 * (len(xs) - 1))], 3),
            "max": round(xs[-1], 3), "n": len(xs)}


def main():
    res = {}
    both = lambda f: (lambda c: (separate(*f(c))[0], merged(*f(c))[0]))  # noqa: E731

    # 1. 缺口检测（有标注）
    gq = e9.gap_question_cases()
    out = pmap(both(lambda c: (c[0],)), gq)
    def pr(pred):
        tp = sum(1 for (q, y), p in zip(gq, pred) if y and p)
        fn = sum(1 for (q, y), p in zip(gq, pred) if y and not p)
        fp = sum(1 for (q, y), p in zip(gq, pred) if not y and p)
        tn = sum(1 for (q, y), p in zip(gq, pred) if not y and not p)
        return {"召回": round(tp / (tp + fn), 4), "误报": round(fp / (fp + tn), 4), "n": len(gq)}
    res["缺口检测"] = {"分开": pr([bool(s["gaps_now"]) for s, m in out]), "合并": pr([bool(m["gaps_now"]) for s, m in out])}
    res["缺口检测_理解模块gaps"] = {"分开": pr([bool(s["gaps"]) for s, m in out]), "合并": pr([bool(m["gaps"]) for s, m in out])}
    agree_intent = sum(1 for s, m in out if s.get("intent") == m.get("intent")) / len(out)
    res["单轮题意图一致率"] = round(agree_intent, 4)

    # 2. 条件抽取（有标注）
    sc = e9.slot_cases()
    out = pmap(both(lambda c: (c[0],)), sc)
    def slots_acc(pick):
        age = sum(1 for (t, g, _), o in zip(sc, out) if (pick(o)["slots"] or {}).get("age") == g["age"]) / len(sc)
        sn = [(g, o) for (t, g, _), o in zip(sc, out) if "has_social" in g]
        soc = sum(1 for g, o in sn if (pick(o)["slots"] or {}).get("has_social") == g["has_social"]) / len(sn)
        return {"年龄": round(age, 4), "社保": round(soc, 4)}
    res["条件抽取"] = {"分开": slots_acc(lambda o: o[0]), "合并": slots_acc(lambda o: o[1])}

    # 3. 身份识别（有标注；用大链路的判定规则）
    ic = e9.identity_cases()
    out = pmap(both(lambda c: (c[0],)), ic)
    def idr(k):
        pos = [(c, o) for c, o in zip(ic, out) if c[1]]
        neg = [(c, o) for c, o in zip(ic, out) if not c[1]]
        return {"召回": round(sum(identity_decision(c[0], o[k]) for c, o in pos) / len(pos), 4),
                "误报": round(sum(identity_decision(c[0], o[k]) for c, o in neg) / len(neg), 4)}
    res["身份识别"] = {"分开": idr(0), "合并": idr(1)}

    # 4. 整通电话语句（无标注，比一致率）
    ct = call_turns()
    out = pmap(lambda r: (separate(r["text"], r["hist"])[0], merged(r["text"], r["hist"])[0]), ct)
    dis = []
    for r, (s, m) in zip(ct, out):
        diff = {k: (s.get(k), m.get(k)) for k in ("act", "intent") if s.get(k) != m.get(k)}
        if set(s["gaps_now"]) != set(m["gaps_now"]):
            diff["gaps_now"] = (s["gaps_now"], m["gaps_now"])
        if set(s["items"]) != set(m["items"]):
            diff["items"] = (s["items"], m["items"])
        if diff:
            dis.append({**r, "diff": diff})
    n = len(ct)
    res["通话语句一致率"] = {"n": n, **{k: round(1 - sum(1 for d in dis if k in d["diff"]) / n, 4)
                                    for k in ("act", "intent", "gaps_now", "items")}}

    # 5. 时延（顺序运行，不并发）
    rng = random.Random(20261008)
    sample = rng.sample(ct, 100)
    ts, tm = [], []
    for i, r in enumerate(sample):
        order = (separate, merged) if i % 2 == 0 else (merged, separate)
        for fn in order:
            _, t = fn(r["text"], r["hist"])
            (ts if fn is separate else tm).append(t)
    res["时延"] = {"分开（三路并行取最慢）": lat(ts), "合并（一次）": lat(tm)}
    res["调用次数每轮"] = {"分开": 3, "合并": 1}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp11_understand_merge.json").write_text(
        json.dumps({"summary": res, "call_disagreements": dis}, ensure_ascii=False, indent=1), encoding="utf-8")
    return res, dis


if __name__ == "__main__":
    r, dis = main()
    print(json.dumps(r, ensure_ascii=False, indent=1))
    print("disagreements", len(dis))
