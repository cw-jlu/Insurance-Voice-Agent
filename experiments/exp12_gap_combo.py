"""实验 13：缺口判断用哪一路、怎么组合。

第六轮整通电话"缺口兜底误触发约 9 通"、80 题全新保留集 13 道错题中 5 道源于缺口误触发
（理解模块把"确诊癌症能直接拿钱吗""没走社保能报多少""胃缩小手术能报吗"判成缺口）。
这里在两批题上比较几种判定：
- U：理解模块（understand）输出的 gaps；
- D：专用缺口检测（small_model.detect_gaps，qwen-flash）；
- J：Jev 缺口概率（阈值 0.7，读 exp10 的缓存，未跑到的题记为缺失）；
- 组合：U∧D、U∨D、J∧U。
题集：第七轮 275 题（qa_v2 + qa_v3 单轮）、新编门槛验证集 88 题（gap_threshold_v1.json）。

用法：python experiments/exp12_gap_combo.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agent import cloud, small_model  # noqa: E402
from agent.pipeline import understand  # noqa: E402

OUT = ROOT / "experiments" / "results"
CLIENT = cloud.llm_client()
JEV_CACHE = json.loads((OUT / "exp10_jev_cache.json").read_text(encoding="utf-8"))
JEV_THR = 0.7


def sets():
    qa = []
    for path in ("qa_v2.json", "qa_v3.json"):
        for it in json.loads((ROOT / "experiments" / "data" / path).read_text(encoding="utf-8")):
            if it.get("valid", True) and not it.get("turns"):
                qa.append((it["question"], "缺口" if it["category"] == "缺口" else None))
    d = json.loads((ROOT / "experiments" / "data" / "gap_threshold_v1.json").read_text(encoding="utf-8"))
    hold = [(q, g) for q, g in d["gap"]] + [(q, None) for q in d["non_gap"]]
    return {"第七轮275题": qa, "新编88题": hold}


def retry(fn, *a):
    for k in range(4):
        try:
            t0 = time.perf_counter()
            return fn(*a), time.perf_counter() - t0
        except Exception:
            time.sleep(2 + 2 * k)
    return None, None


def one(q):
    u, tu = retry(lambda: understand(q, {}, client=CLIENT))
    d, td = retry(lambda: small_model.detect_gaps(q, CLIENT))
    key = json.dumps(["jev_gaps", "jev-1.13-free", q], ensure_ascii=False)
    j = JEV_CACHE.get(key, [None])[0]
    return {"q": q, "U": (u or {}).get("gaps", []), "D": d or [], "J": None if j is None else [g for g, p in j.items() if p >= JEV_THR],
            "t_understand": tu, "t_detect": td}


def metrics(rows, labels, pick):
    pos = [pick(r) for r, y in zip(rows, labels) if y]
    neg = [pick(r) for r, y in zip(rows, labels) if not y]
    pos, neg = [p for p in pos if p is not None], [p for p in neg if p is not None]
    return {"召回": f"{sum(bool(p) for p in pos)}/{len(pos)}", "误报": f"{sum(bool(p) for p in neg)}/{len(neg)}"}


COMBOS = {
    "U 理解模块": lambda r: r["U"],
    "D 专用检测(qwen)": lambda r: r["D"],
    "U∧D": lambda r: r["D"] if (r["U"] and r["D"]) else [],
    "U∨D": lambda r: r["U"] or r["D"],
    "J Jev≥0.7": lambda r: r["J"],
    "J∧U": lambda r: None if r["J"] is None else (r["J"] if (r["J"] and r["U"]) else []),
}


if __name__ == "__main__":
    res = {}
    for name, items in sets().items():
        with ThreadPoolExecutor(8) as ex:
            rows = list(ex.map(lambda it: one(it[0]), items))
        labels = [y is not None for _, y in items]
        res[name] = {k: metrics(rows, labels, f) for k, f in COMBOS.items()}
        res[name]["rows"] = [{**r, "y": y} for r, (_, y) in zip(rows, items)]
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "rows"}, ensure_ascii=False))
    (OUT / "exp12_gap_combo.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
