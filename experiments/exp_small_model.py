"""第四轮：小模型替代纯规则。在第三轮的扩大测试集上，对比"规则"与"规则快路径 + 小模型"。

用例与第三轮完全相同（同一随机种子生成），小模型提示词只描述任务、不引用测试用例。
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

from agent import cloud, small_model  # noqa: E402
from agent.dialogue import classify_barge_in as rule_barge  # noqa: E402
from agent.guardrail import gap_guard, is_identity_question as rule_identity  # noqa: E402
from agent.speech import extract_slots as rule_slots  # noqa: E402

OUT = ROOT / "experiments" / "results"
CLIENT = cloud.llm_client()


def timed(fn, *a):
    for attempt in range(3):
        try:
            t0 = time.perf_counter()
            r = fn(*a, client=CLIENT)
            return r, time.perf_counter() - t0
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def lat(xs):
    xs = sorted(xs)
    return {"p50": round(statistics.median(xs), 3), "p90": round(xs[int(0.9 * (len(xs) - 1))], 3)} if xs else None


def barge():
    import exp5_interrupt_v2 as e5
    e5.rng.seed(20261005)
    cases = e5.build_5a()
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda c: timed(small_model.classify_barge_in, c[0]), cases))
    rows = []
    for (u, exp), ((label, path), t) in zip(cases, res):
        rows.append({"u": u, "exp": exp, "rule": rule_barge(u), "hybrid": "backchannel" if label == "附和" else "interrupt",
                     "path": path, "t": t})
    bc = [r for r in rows if r["exp"] == "backchannel"]
    it = [r for r in rows if r["exp"] == "interrupt"]
    model_t = [r["t"] for r in rows if r["path"] == "model"]
    return {
        "cases": len(rows),
        "rule": {"false_stop": round(sum(r["rule"] != "backchannel" for r in bc) / len(bc), 4),
                 "interrupt_recall": round(sum(r["rule"] == "interrupt" for r in it) / len(it), 4)},
        "rule+model": {"false_stop": round(sum(r["hybrid"] != "backchannel" for r in bc) / len(bc), 4),
                       "interrupt_recall": round(sum(r["hybrid"] == "interrupt" for r in it) / len(it), 4)},
        "model_path_share": round(len(model_t) / len(rows), 4),
        "model_latency_s": lat(model_t),
        "errors": [r for r in rows if r["hybrid"] != r["exp"]][:15],
    }


def slots():
    import exp6_slots_v2 as e6
    cases = e6.build()
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda c: timed(small_model.extract_slots, c[0]), cases))
    out = {}
    for name, getter in (("rule", lambda i: rule_slots(cases[i][0])), ("model", lambda i: res[i][0])):
        per, exact, errs = {}, 0, []
        for i, (text, gold, _) in enumerate(cases):
            got = getter(i)
            ok = True
            for k, v in gold.items():
                c = per.setdefault(k, [0, 0])
                c[1] += 1
                if got.get(k) == v:
                    c[0] += 1
                else:
                    ok = False
            exact += ok
            if not ok and len(errs) < 12:
                errs.append({"text": text, "gold": gold, "got": got})
        out[name] = {"all_correct": round(exact / len(cases), 4), "per_slot": {k: round(c / n, 4) for k, (c, n) in per.items()},
                     "errors": errs if name == "model" else []}
    out["model_latency_s"] = lat([t for _, t in res])
    out["cases"] = len(cases)
    return out


def identity():
    import exp3_guardrail_v2 as e3
    e3.rng.seed(20261007)
    e3.correct_statements(); e3.run_gap(); e3.run_compliance()  # advance the RNG exactly as in round 3
    rng = e3.rng
    pos = []
    for _ in range(120):
        t = rng.choice(e3.ID_TERM)
        pos.append(rng.choice(["", "喂，", "哎，", "我问一下，"]) + rng.choice(e3.ID_SUBJ) + rng.choice(e3.ID_ASK).format(t=t))
    neg = e3.ID_NEG * 3
    cases = [(u, True) for u in pos] + [(u, False) for u in neg]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda c: timed(small_model.is_identity_question, c[0]), cases))
    def score(pred):
        tp = sum(p and y for p, (_, y) in zip(pred, cases))
        fp = sum(p and not y for p, (_, y) in zip(pred, cases))
        return {"recall": round(tp / len(pos), 4), "false_alarm": round(fp / len(neg), 4)}
    model_pred = [r for r, _ in res]
    return {"positives": len(pos), "negatives": len(neg), "rule": score([rule_identity(u) for u, _ in cases]),
            "model": score(model_pred), "rule_or_model": score([rule_identity(u) or m for (u, _), m in zip(cases, model_pred)]),
            "model_latency_s": lat([t for _, t in res]),
            "model_errors": [u for (u, y), p in zip(cases, model_pred) if p != y][:15]}


def gaps():
    import exp3_guardrail_v2 as e3
    e3.rng.seed(20261007)
    e3.correct_statements()
    rng = e3.rng
    cases = []
    for _ in range(150):
        d = rng.choice(e3.DISEASES)
        q = rng.choice(e3.Q_UW).format(d=d)
        bad = rng.random() < 0.5
        cases.append((q, rng.choice(e3.A_UW_BAD if bad else e3.A_UW_OK).format(d=d), bad))
    for _ in range(50):
        bad = rng.random() < 0.5
        cases.append((rng.choice(e3.Q_RN), rng.choice(e3.A_RN_BAD if bad else e3.A_RN_OK), bad))
    for _ in range(30):
        bad = rng.random() < 0.5
        cases.append((rng.choice(e3.Q_DD), rng.choice(e3.A_DD_BAD if bad else e3.A_DD_OK), bad))
    for _ in range(40):
        cases.append((rng.choice(e3.Q_NON), rng.choice(e3.A_NON), False))
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda c: timed(small_model.gap_review, c[0], c[1]), cases))

    def score(pred):
        tp = sum(p and y for p, (_, _, y) in zip(pred, cases))
        fn = sum((not p) and y for p, (_, _, y) in zip(pred, cases))
        fp = sum(p and not y for p, (_, _, y) in zip(pred, cases))
        tn = sum((not p) and not y for p, (_, _, y) in zip(pred, cases))
        return {"recall": round(tp / (tp + fn), 4), "false_block": round(fp / (fp + tn), 4)}
    rule = [not gap_guard(q, a).ok for q, a, _ in cases]
    model = [r for r, _ in res]
    return {"cases": len(cases), "rule": score(rule), "model": score(model), "rule_or_model": score([a or b for a, b in zip(rule, model)]),
            "model_latency_s": lat([t for _, t in res]),
            "model_errors": [{"q": q, "a": a, "expected": y} for (q, a, y), p in zip(cases, model) if p != y][:12]}


def main():
    res = {"model": cloud.FAST_MODEL, "barge_in": barge(), "slots": slots(), "identity": identity(), "gap_review": gaps()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp_small_model.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main()
    for k in ("barge_in", "slots", "identity", "gap_review"):
        print(k, json.dumps({x: y for x, y in r[k].items() if x not in ("errors", "model_errors")}, ensure_ascii=False))
