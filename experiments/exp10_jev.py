"""实验 12：Jev（OpenCode Zen 的 System One 分类模型）替代小模型的分类环节。

Jev 不生成文本，只回答带类型的问题并返回概率，适合"是/否""多选一"的环节：
打断、身份识别、缺口检测、缺口复核、对话作用（含同意判断）。条件抽取和事实核对要生成结构化文本，不适用。

与第七轮（exp9_models.py small）用同一批标注集；qwen-flash 的结果直接读 exp9_models_small.json。
因为 Jev 给出概率，额外报告阈值扫描。同意判断另用一组新编的短句（明确同意 / 犹豫 / 拒绝）。

免费版额度很小（第一次运行约 20 次调用后即被限流，retry-after 约 7.7 小时），所以每次调用的结果都缓存到
results/exp10_jev_cache.json；遇到长时间限流就保存并退出，额度恢复后重跑同一命令即从断点继续。

用法：python experiments/exp10_jev.py [model]   # 默认 jev-1.13-free
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agent import cloud, small_model  # noqa: E402
from agent.dialogue import _BACKCHANNEL, _STOP  # noqa: E402
from exp9_models import CLIENT, barge_cases, gap_question_cases, gap_review_cases, identity_cases, lat  # noqa: E402

OUT = ROOT / "experiments" / "results"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# 同意判断（第六轮 P0：2 通未经同意发链接）。坐席刚问过"给您发投保链接可以吗"。
CONSENT_YES = ["发吧", "发过来", "可以发", "行，发过来吧", "好的你发我手机上", "嗯，发吧", "那你发一个链接给我",
               "可以，我看看怎么买", "行吧，我买一份", "好，给我发个链接", "发我短信就行", "没问题，发吧",
               "那就买这个吧", "可以可以，发过来", "好啊，你把链接发我", "就它了，怎么投保"]
CONSENT_NO = ["我再考虑考虑", "我想想再说", "回头再说吧", "先挂了，我跟老婆商量一下", "我看看再说", "不用了",
              "先别发", "我还没想好", "等我有空再看", "嗯……我再比较一下别家", "太贵了，算了", "我先问问家里人",
              "发什么发，我不需要", "过两天再说吧", "这个我得考虑一下", "好的，我知道了，我再想想"]


def pmap(fn, items):
    """Sequential on purpose: 8 parallel calls tripped the free tier's limit (7.7 h lockout) after ~20 calls."""
    out = []
    for n, it in enumerate(items, 1):
        out.append(fn(it))
        if n % 20 == 0:
            save_cache()
    return out


CACHE_PATH = OUT / "exp10_jev_cache.json"
_cache: dict = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
_cache_lock = threading.Lock()


class QuotaExhausted(Exception):
    pass


def save_cache():
    with _cache_lock:
        CACHE_PATH.write_text(json.dumps(_cache, ensure_ascii=False), encoding="utf-8")


def timed(fn, *a):
    """Cached Jev call → (result, seconds). Raises QuotaExhausted on a long rate-limit window."""
    key = json.dumps([fn.__name__, cloud.JEV_MODEL, *(x for x in a if isinstance(x, str))], ensure_ascii=False)
    if key in _cache:
        return tuple(_cache[key])
    for attempt in range(4):
        try:
            t0 = time.perf_counter()
            r = fn(*a)
            out = (r, time.perf_counter() - t0)
            with _cache_lock:
                _cache[key] = out
            return out
        except Exception as e:
            if "429" in str(e) and "FreeUsageLimit" in str(e):
                raise QuotaExhausted(str(e)) from e
            if attempt == 3:
                print("fail:", e)
                return None, None
            time.sleep(2 + 3 * attempt)


def binary(probs, labels, thr):
    pos = [p for p, y in zip(probs, labels) if y and p is not None]
    neg = [p for p, y in zip(probs, labels) if not y and p is not None]
    rate = lambda xs: round(sum(p >= thr for p in xs) / len(xs), 4) if xs else None
    return {"召回": rate(pos), "误报": rate(neg)}


def sweep(probs, labels):
    return {str(t): binary(probs, labels, t) for t in THRESHOLDS}


def task_consent(res):
    """同意判断：Jev 的"同意投保"概率 vs qwen-flash 的 classify_act。"""
    hist = "坐席：好的，那我把投保链接发到您手机上，您看可以吗？"
    utt = CONSENT_YES + CONSENT_NO
    labels = [True] * len(CONSENT_YES) + [False] * len(CONSENT_NO)
    out = pmap(lambda u: timed(small_model.jev_act, u, hist), utt)
    jp = [o[0]["probabilities"].get("同意投保", 0.0) if o[0] else None for o in out]
    jc = [o[0]["choice"] == "同意投保" if o[0] else None for o in out]
    small_model.JEV_TASKS.discard("act")
    cloud.FAST_MODEL = "qwen-flash"
    qo = pmap(lambda u: timed(small_model.classify_act, u, hist, CLIENT), utt)
    qc = [o[0] == "同意投保" if o[0] else None for o in qo]

    def cm(pred):
        return {"明确同意识别": f"{sum(1 for p, y in zip(pred, labels) if y and p)}/{len(CONSENT_YES)}",
                "犹豫/拒绝误判为同意": f"{sum(1 for p, y in zip(pred, labels) if not y and p)}/{len(CONSENT_NO)}"}

    res["同意判断"] = {"jev_choice": cm(jc), "jev_概率阈值": {str(t): cm([p is not None and p >= t for p in jp]) for t in (0.5, 0.7, 0.9)},
                   "qwen-flash": cm(qc), "jev_时延": lat([o[1] for o in out]), "qwen_时延": lat([o[1] for o in qo]),
                   "rows": [{"u": u, "y": y, "jev": o[0], "qwen": q[0]} for u, y, o, q in zip(utt, labels, out, qo)]}


def task_gaps(res):
    gq = gap_question_cases()
    out = pmap(lambda c: timed(small_model.jev_gaps, c[0]), gq)
    probs = [max(o[0].values()) if o[0] else None for o in out]
    s = sweep(probs, [y for _, y in gq])
    res["缺口检测"] = {**s["0.5"], "n": len(gq), "阈值扫描": s, "时延": lat([o[1] for o in out]), "失败": sum(p is None for p in probs),
                   "rows": [{"q": q, "y": y, "p": o[0]} for (q, y), o in zip(gq, out)]}


def task_gap_review(res):
    gr = gap_review_cases()
    out = pmap(lambda c: timed(small_model.jev_gap_review, c[0], c[1]), gr)
    s = sweep([o[0] for o in out], [c[2] for c in gr])
    res["缺口复核"] = {"召回": s["0.5"]["召回"], "误拦": s["0.5"]["误报"], "n": len(gr), "阈值扫描": s,
                   "时延": lat([o[1] for o in out]), "失败": sum(o[0] is None for o in out)}


def task_identity(res):
    ic = identity_cases()
    out = pmap(lambda c: timed(small_model.jev_identity, c[0]), ic)
    s = sweep([o[0] for o in out], [y for _, y in ic])
    res["身份识别"] = {**s["0.5"], "n": len(ic), "阈值扫描": s, "时延": lat([o[1] for o in out]), "失败": sum(o[0] is None for o in out)}


def task_barge(res):
    """打断：规则快路径与第七轮相同，只有规则判不了的句子交给模型。"""
    bc = barge_cases()
    need = [i for i, c in enumerate(bc) if c[0].strip() and not _BACKCHANNEL.match(c[0].strip()) and not _STOP.search(c[0])]
    out = pmap(lambda i: timed(small_model.jev_barge_in, bc[i][0]), need)
    p_of = {i: o[0] for i, o in zip(need, out)}
    probs = [p_of.get(i, 0.0 if (not c[0].strip() or _BACKCHANNEL.match(c[0].strip())) else 1.0) for i, c in enumerate(bc)]
    labels = [c[1] == "interrupt" for c in bc]
    s = sweep(probs, labels)
    res["打断"] = {"附和误判为打断": s["0.5"]["误报"], "真打断识别": s["0.5"]["召回"], "模型路径句数": len(need),
                 "阈值扫描": s, "时延": lat([o[1] for o in out]), "失败": sum(o[0] is None for o in out)}


def _holdout():
    d = json.loads((ROOT / "experiments" / "data" / "gap_threshold_v1.json").read_text(encoding="utf-8"))
    return [(q, g) for q, g in d["gap"]] + [(q, None) for q in d["non_gap"]]


def _holdout_metrics(preds):
    """preds: list of detected gap lists, aligned with _holdout()."""
    hc = _holdout()
    pos = [(g, p) for (_, g), p in zip(hc, preds) if g and p is not None]
    neg = [p for (_, g), p in zip(hc, preds) if not g and p is not None]
    return {"召回": f"{sum(bool(p) for _, p in pos)}/{len(pos)}", "类型正确": f"{sum(g in p for g, p in pos)}/{len(pos)}",
            "误报": f"{sum(bool(p) for p in neg)}/{len(neg)}"}


def task_gap_holdout_qwen(res):
    """门槛验证集上的 qwen-flash 基线（不耗 Jev 额度）。"""
    hc = _holdout()
    cloud.FAST_MODEL = "qwen-flash"
    out = pmap(lambda c: timed(small_model.detect_gaps, c[0], CLIENT), hc)
    res["缺口检测·新题·qwen-flash"] = {**_holdout_metrics([o[0] for o in out]), "n": len(hc), "时延": lat([o[1] for o in out]),
                                  "rows": [{"q": q, "y": g, "pred": o[0]} for (q, g), o in zip(hc, out)]}


def task_gap_holdout_jev(res):
    """门槛 0.7 是在第七轮题上看出来的，在新编的题上验证。"""
    hc = _holdout()
    out = pmap(lambda c: timed(small_model.jev_gaps, c[0]), hc)
    res["缺口检测·新题·Jev"] = {
        **{f"阈值{t}": _holdout_metrics([None if o[0] is None else [g for g, p in o[0].items() if p >= t] for o in out])
           for t in (0.5, 0.6, 0.7, 0.8)},
        "n": len(hc), "时延": lat([o[1] for o in out]), "失败": sum(o[0] is None for o in out),
        "rows": [{"q": q, "y": g, "p": o[0]} for (q, g), o in zip(hc, out)]}


# most decision-relevant first, so a partial run (quota) still answers the main questions
TASKS = [task_consent, task_gaps, task_gap_holdout_qwen, task_gap_review, task_gap_holdout_jev, task_identity, task_barge]


def run(model: str) -> tuple[dict, str | None]:
    cloud.JEV_MODEL = model
    prev = OUT / "exp10_jev.json"
    res: dict = {k: v for k, v in json.loads(prev.read_text(encoding="utf-8")).items() if k not in ("model", "stopped")}         if prev.exists() else {}
    for task in TASKS:
        try:
            task(res)
        except QuotaExhausted as e:
            return res, f"{task.__name__}: {e}"
        save_cache()
        print(task.__name__, "done", flush=True)
    return res, None


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "jev-1.13-free"
    try:
        res, stopped = run(model)
    finally:
        save_cache()
    (OUT / "exp10_jev.json").write_text(json.dumps({"model": model, "stopped": stopped, **res}, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    for k, v in res.items():
        print(k, json.dumps({a: b for a, b in v.items() if a != "rows"}, ensure_ascii=False))
    if stopped:
        sys.exit(f"Jev 额度用尽，已缓存 {len(_cache)} 次调用，额度恢复后重跑即可续上（{stopped}）")
