"""实验 11：模型对比（qwen vs DeepSeek）。

A. 小模型环节：每个环节都用已有的标注集打分，比较准确率和单次调用时延。
   打断判断（240 句）、条件抽取（200 句）、缺口检测（问答测试集中的 289 题，类别为"缺口"的为正例）、
   身份识别（第三轮 156 例 + 新编 20 例）、缺口复核（270 + 180 例）、
   事实核对（80 题全新保留集的回答，以人工复核结论为标签：错误回答应被拦下）。
B. 回答生成：80 题全新保留集，理解等环节固定用 qwen-flash，只换生成模型。

用法：python experiments/exp9_models.py small | gen
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agent import cloud, small_model  # noqa: E402

OUT = ROOT / "experiments" / "results"
SMALL_MODELS = ["qwen-flash", "deepseek-v4-flash", "deepseek-v4.1-flash", "deepseek-v3.2"]
GEN_MODELS = ["qwen-plus", "deepseek-v4-flash", "deepseek-v4-pro"]
CLIENT = cloud.llm_client()

# 80 题全新保留集上人工复核为错误的题（第六轮，见 06 实验记录 二十五）
WRONG_V3 = {"N006", "N010", "N012", "N017", "N020", "N037", "N038", "N042", "N054", "N063", "N064", "N076", "N079"}


def timed(fn, *a):
    for attempt in range(3):
        try:
            t0 = time.perf_counter()
            r = fn(*a)
            return r, time.perf_counter() - t0
        except Exception:
            if attempt == 2:
                return None, None
            time.sleep(2 + 2 * attempt)


def lat(xs):
    xs = sorted(x for x in xs if x is not None)
    return {"p50": round(statistics.median(xs), 3), "p90": round(xs[int(0.9 * (len(xs) - 1))], 3)} if xs else None


def pmap(fn, items, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


# ---------------------------------------------------------------- task sets
def barge_cases():
    import exp5_interrupt_v2 as e5
    e5.rng.seed(20261005)
    return e5.build_5a()


def slot_cases():
    import exp6_slots_v2 as e6
    return e6.build()


def gap_question_cases():
    rows = []
    for path in ("qa_v2.json", "qa_v3.json"):
        for it in json.loads((ROOT / "experiments" / "data" / path).read_text(encoding="utf-8")):
            if it.get("valid", True) and not it.get("turns"):
                rows.append((it["question"], it["category"] == "缺口"))
    return rows


def identity_cases():
    import exp3_guardrail_v2 as e3
    import exp_small_model_fresh as fr
    e3.rng.seed(20261007)
    e3.correct_statements(); e3.run_gap(); e3.run_compliance()
    pos = []
    for _ in range(120):
        t = e3.rng.choice(e3.ID_TERM)
        pos.append(e3.rng.choice(["", "喂，", "哎，", "我问一下，"]) + e3.rng.choice(e3.ID_SUBJ) + e3.rng.choice(e3.ID_ASK).format(t=t))
    return [(u, True) for u in pos + fr.IDP] + [(u, False) for u in e3.ID_NEG * 3 + fr.IDN]


def gap_review_cases():
    import exp3_guardrail_v2 as e3
    import exp_small_model_fresh as fr
    e3.rng.seed(20261007)
    e3.correct_statements()
    rng = e3.rng
    cases = []
    for _ in range(150):
        d = rng.choice(e3.DISEASES)
        bad = rng.random() < 0.5
        cases.append((rng.choice(e3.Q_UW).format(d=d), rng.choice(e3.A_UW_BAD if bad else e3.A_UW_OK).format(d=d), bad))
    for _ in range(50):
        bad = rng.random() < 0.5
        cases.append((rng.choice(e3.Q_RN), rng.choice(e3.A_RN_BAD if bad else e3.A_RN_OK), bad))
    for _ in range(30):
        bad = rng.random() < 0.5
        cases.append((rng.choice(e3.Q_DD), rng.choice(e3.A_DD_BAD if bad else e3.A_DD_OK), bad))
    # fresh set (only gap questions: the review runs only on questions judged as gaps)
    fr.rng.seed(20261008)
    for _ in range(120):
        d = fr.rng.choice(fr.DIS)
        bad = fr.rng.random() < 0.5
        cases.append((fr.rng.choice(fr.Q).format(d=d), fr.rng.choice(fr.BAD if bad else fr.OK).format(d=d), bad))
    for _ in range(30):
        bad = fr.rng.random() < 0.5
        cases.append((fr.rng.choice(fr.RQ), fr.rng.choice(fr.RBAD if bad else fr.ROK), bad))
    return cases


def fact_cases():
    """80 answers + the evidence the pipeline would retrieve for each question (built once, same for all models)."""
    cache = OUT / "exp9_fact_evidence.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from agent.pipeline import HybridRetriever, Pipeline, grouped_context, item_chunks, topic_chunks, understand
    from embed_cache import corpus, load
    chunks = corpus(False)
    ret = HybridRetriever(chunks, load(chunks, "clauses"), client=CLIENT)
    rows = json.loads((OUT / "exp2_qa_v2_v3.json").read_text(encoding="utf-8"))["rows"]

    def build(r):
        u = understand(r["question"], {}, client=CLIENT)
        ev = []
        for i in u.get("items", []):
            ev += item_chunks(i)
        for t in u.get("topics", []):
            if t != "家庭单优惠":
                ev += topic_chunks(t)
        ev += ret.search(u.get("rewrite") or r["question"], k=4)
        seen, uniq = set(), []
        for c in ev:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                uniq.append(c)
        return {"id": r["id"], "answer": r["pipeline"]["text"], "evidence": grouped_context(uniq), "wrong": r["id"] in WRONG_V3}

    out = pmap(build, rows)
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------- scoring
def run_small(model: str) -> dict:
    cloud.FAST_MODEL = model
    res = {}

    bc = barge_cases()
    out = pmap(lambda c: timed(small_model.classify_barge_in, c[0], CLIENT), bc)
    model_path = [(c, o) for c, o in zip(bc, out) if o[0] and o[0][1] == "model"]
    bc_pos = [(c, o) for c, o in zip(bc, out) if c[1] == "backchannel"]
    it_pos = [(c, o) for c, o in zip(bc, out) if c[1] == "interrupt"]
    res["打断"] = {"附和误判为打断": round(sum(1 for c, o in bc_pos if not o[0] or o[0][0] != "附和") / len(bc_pos), 4),
                 "真打断识别": round(sum(1 for c, o in it_pos if o[0] and o[0][0] == "打断") / len(it_pos), 4),
                 "时延": lat([o[1] for c, o in model_path])}

    sc = slot_cases()
    out = pmap(lambda c: timed(small_model.extract_slots, c[0], CLIENT), sc)
    age = sum(1 for (t, g, _), (r, _) in zip(sc, out) if r and r.get("age") == g["age"]) / len(sc)
    soc_n = [((t, g), r) for (t, g, _), (r, _) in zip(sc, out) if "has_social" in g]
    soc = sum(1 for (t, g), r in soc_n if r and r.get("has_social") == g["has_social"]) / len(soc_n)
    res["条件抽取"] = {"年龄": round(age, 4), "社保": round(soc, 4), "时延": lat([o[1] for o in out])}

    gq = gap_question_cases()
    out = pmap(lambda c: timed(small_model.detect_gaps, c[0], CLIENT), gq)
    tp = sum(1 for (q, y), (r, _) in zip(gq, out) if y and r)
    fn = sum(1 for (q, y), (r, _) in zip(gq, out) if y and not r)
    fp = sum(1 for (q, y), (r, _) in zip(gq, out) if not y and r)
    tn = sum(1 for (q, y), (r, _) in zip(gq, out) if not y and not r)
    res["缺口检测"] = {"召回": round(tp / (tp + fn), 4), "误报": round(fp / (fp + tn), 4), "n": len(gq), "时延": lat([o[1] for o in out])}

    ic = identity_cases()
    out = pmap(lambda c: timed(small_model.is_identity_question, c[0], CLIENT), ic)
    pos = [(c, o) for c, o in zip(ic, out) if c[1]]
    neg = [(c, o) for c, o in zip(ic, out) if not c[1]]
    res["身份识别"] = {"召回": round(sum(1 for c, o in pos if o[0]) / len(pos), 4),
                   "误报": round(sum(1 for c, o in neg if o[0]) / len(neg), 4), "时延": lat([o[1] for o in out])}

    gr = gap_review_cases()
    out = pmap(lambda c: timed(small_model.gap_review, c[0], c[1], CLIENT), gr)
    tp = sum(1 for c, o in zip(gr, out) if c[2] and o[0])
    fn = sum(1 for c, o in zip(gr, out) if c[2] and not o[0])
    fp = sum(1 for c, o in zip(gr, out) if not c[2] and o[0])
    tn = sum(1 for c, o in zip(gr, out) if not c[2] and not o[0])
    res["缺口复核"] = {"召回": round(tp / (tp + fn), 4), "误拦": round(fp / (fp + tn), 4), "n": len(gr), "时延": lat([o[1] for o in out])}

    fc = fact_cases()
    out = pmap(lambda c: timed(small_model.fact_check, c["answer"], c["evidence"], CLIENT), fc)
    wrong = [(c, o) for c, o in zip(fc, out) if c["wrong"]]
    right = [(c, o) for c, o in zip(fc, out) if not c["wrong"]]
    res["事实核对"] = {"错误回答被拦下": f"{sum(1 for c, o in wrong if o[0])}/{len(wrong)}",
                   "正确回答被误拦": f"{sum(1 for c, o in right if o[0])}/{len(right)}", "时延": lat([o[1] for o in out])}
    return res


def run_gen(model: str) -> dict:
    from agent.pipeline import HybridRetriever, Pipeline
    from embed_cache import corpus, load
    import exp2_qa_v2 as e2

    chunks = corpus(False)
    pipe = Pipeline(HybridRetriever(chunks, load(chunks, "clauses"), client=CLIENT), client=CLIENT, slow_model=model,
                    fast_model="qwen-flash")
    items = json.loads((ROOT / "experiments" / "data" / "qa_v3.json").read_text(encoding="utf-8"))

    def one(it):
        known = dict(it.get("context") or {})
        if it.get("turns"):
            u = e2.retry(lambda: e2.understand(it["turns"][0], known, client=CLIENT))
            known.update({k: v for k, v in (u.get("slots") or {}).items() if v is not None})
            known["_prev_utterance"], known["_prev_intent"] = it["turns"][0], u.get("intent")
        t0 = time.perf_counter()
        a = e2.retry(lambda: pipe.answer(it["question"], known))
        latency = time.perf_counter() - t0
        rx, miss, forb = e2.regex_judge(it, a.text)
        jd, reason = e2.retry(lambda: e2.model_judge(it, a.text, CLIENT))
        return {"id": it["id"], "category": it["category"], "question": it["question"], "text": a.text,
                "regex": rx, "judge": jd, "reason": reason, "latency": round(latency, 3),
                "ttft": round(a.timings.get("generate_ttft", 0), 3), "attempts": a.attempts, "fallback": a.fallback_used}

    rows = pmap(one, items, workers=6)
    n = len(rows)
    lats = sorted(r["latency"] for r in rows)
    return {"model": model, "n": n, "regex": round(sum(r["regex"] for r in rows) / n, 4),
            "judge_deepseek": round(sum(r["judge"] for r in rows) / n, 4),
            "both": round(sum(r["regex"] and r["judge"] for r in rows) / n, 4),
            "fallback": sum(r["fallback"] for r in rows), "retries": sum(r["attempts"] > 1 for r in rows),
            "latency_p50": round(statistics.median(lats), 2), "latency_p90": round(lats[int(0.9 * (n - 1))], 2),
            "ttft_p50": round(statistics.median(r["ttft"] for r in rows), 2), "rows": rows}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "small"
    if mode == "small":
        res = {m: run_small(m) for m in SMALL_MODELS}
        (OUT / "exp9_models_small.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        for m, r in res.items():
            print(m, json.dumps(r, ensure_ascii=False))
    else:
        res = [run_gen(m) for m in GEN_MODELS]
        (OUT / "exp9_models_gen.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        for r in res:
            print({k: v for k, v in r.items() if k != "rows"})
