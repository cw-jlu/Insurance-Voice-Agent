"""Offline replay: independent small-model evidence gate versus the old keyword gate.

The question's reference facts provide required coverage IDs. For each question we
test a complete set, a set missing one required ID, and a set with a distracting ID
in its place. This tests responsibility coverage, not clause or answer correctness.

Run from the repository root:
    python experiments/exp16_evidence_gate.py --env-file PATH_TO_ENV
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from agent import cloud  # noqa: E402
from agent.knowledge import coverage_items  # noqa: E402
from data_complex import items as complex_items  # noqa: E402

OLD_BRANCH = "claude/review-summary-table-expand-tests-8a5cd8"
OUT_DIR = ROOT / "experiments" / "results"
IDS = re.compile(r"R\d{2}")
EXTRA = [
    ("重症监护室一天给多少钱，最多给几天？", ["R15"], ["R01"]),
    ("白血病治疗费免赔吗，最多报多少？", ["R03"], ["R01"]),
    ("附加的普通门诊保障多大能买，一次最多报多少？", ["R13"], ["R02"]),
    ("开刀后请人照顾能补贴吗，一天多少钱？", ["R08"], ["R14"]),
    ("免疫治疗药能报销吗，最多赔多少？", ["R06"], ["R03"]),
    ("家人一起买免赔额合并计算吗，怎么算？", ["R09"], ["R01"]),
    ("做粒子治疗能报吗，最多多少钱？", ["R05"], ["R03"]),
]
SIMILAR = {"R01": "R03", "R02": "R13", "R03": "R01", "R04": "R01", "R05": "R06",
           "R06": "R05", "R07": "R01", "R08": "R14", "R09": "R01", "R10": "R03",
           "R11": "R01", "R12": "R01", "R13": "R02", "R14": "R15", "R15": "R14", "R16": "R06"}


def old_file(path: str) -> str:
    return subprocess.run(["git", "show", f"{OLD_BRANCH}:{path}"], cwd=ROOT,
                          check=True, capture_output=True, text=True, encoding="utf-8").stdout


def keyword_patterns() -> list[tuple[str, list[str]]]:
    tree = ast.parse(old_file("experiments/exp12_agent_workflow.py"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KEYWORDS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("old KEYWORDS table not found")


def questions() -> list[dict]:
    sources = [(f"qa_v{i}", json.loads((ROOT / "experiments" / "data" / f"qa_v{i}.json").read_text(encoding="utf-8")))
               for i in range(2, 8)]
    sources += [("old_qa_v4", json.loads(old_file("experiments/data/qa_v4.json"))),
                ("complex", complex_items())]
    seen: set[str] = set()
    out: list[dict] = []
    for source, rows in sources:
        for row in rows:
            if (row.get("valid") is False or row.get("turns") or row.get("category") == "价格"
                    or not (gold := sorted(set(IDS.findall(row.get("facts", "")))))):
                continue
            q = row["question"]
            if q in seen:
                continue
            seen.add(q)
            out.append({"id": f"{source}:{row['id']}", "question": q, "gold": gold, "source": source})
    for n, (q, gold, distractors) in enumerate(EXTRA, 1):
        out.append({"id": f"paraphrase:{n}", "question": q, "gold": gold,
                    "source": "paraphrase", "distractors": distractors})
    return out


def negative_questions() -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    sources = [(f"qa_v{i}", json.loads((ROOT / "experiments" / "data" / f"qa_v{i}.json").read_text(encoding="utf-8")))
               for i in range(2, 8)]
    sources.append(("old_qa_v4", json.loads(old_file("experiments/data/qa_v4.json"))))
    for source, rows in sources:
        for row in rows:
            if row.get("valid") is False or row.get("turns") or row.get("category") not in ("价格", "优惠", "身份"):
                continue
            q = row["question"]
            if q in seen:
                continue
            seen.add(q)
            out.append({"id": f"{source}:{row['id']}", "question": q, "gold": [],
                        "source": source, "category": row["category"]})
    return out


def ask(client, row: dict, catalog: str) -> dict:
    prompt = ("你是保险问答的取证检查员。只列出回答这个问题不可缺少的保障责任，最多5项。"
              "只看用户明确问到的项目，不列相关但未问的责任，不列所有可能适用的责任。"
              "普通住院、重疾住院、普通门诊、门诊加油包、一般津贴、ICU津贴分别是不同责任。"
              "只问保费或加油包价格时，责任列表留空；只问医院资格、职业、免责等条款时也留空。"
              "一句话有多个具体问题时逐个考虑。只输出JSON："
              "{\"required_items\":[\"R01\"],\"topics\":[\"医院范围\"]}。"
              "topics只列需要查的关键条款名称。\n责任目录：\n" + catalog)
    t0 = time.perf_counter()
    try:
        raw = cloud.chat([{"role": "system", "content": prompt},
                          {"role": "user", "content": row["question"]}],
                         model=cloud.FAST_MODEL, client=client, temperature=0,
                         response_format={"type": "json_object"})
        data = json.loads(raw)
        predicted = sorted({x for x in data.get("required_items", []) if isinstance(x, str) and IDS.fullmatch(x)})
        return {"predicted": predicted, "topics": data.get("topics", []),
                "latency": round(time.perf_counter() - t0, 3), "error": None}
    except Exception as exc:
        return {"predicted": None, "topics": [], "latency": round(time.perf_counter() - t0, 3),
                "error": type(exc).__name__}


def ask_direct(client, row: dict, have: set[str], catalog: str) -> dict:
    prompt = ("判断现有保障责任资料是否覆盖用户明确问到的每项保障责任。只检查责任种类是否齐，"
              "暂不检查条款细节或答案是否正确。若缺少，列出缺失的责任编号；基础版和加油包、"
              "一般住院和重疾住院、普通门诊和门诊加油包、一般住院津贴和ICU津贴不可互相替代。"
              "只问保费时无需保障责任。已有责任编号绝不能列为缺失；只列没有给你的编号。"
              "最多列5项。只输出JSON：{\"missing_items\":[\"R01\"]}。"
              "\n责任目录：\n" + catalog)
    user = f"用户问题：{row['question']}\n已有责任编号：{','.join(sorted(have)) or '无'}"
    t0 = time.perf_counter()
    try:
        raw = cloud.chat([{"role": "system", "content": prompt}, {"role": "user", "content": user}],
                         model=cloud.FAST_MODEL, client=client, temperature=0,
                         response_format={"type": "json_object"})
        data = json.loads(raw)
        missing = sorted({x for x in data.get("missing_items", [])
                          if isinstance(x, str) and IDS.fullmatch(x) and x not in have})
        return {"trigger": bool(missing), "missing_items": missing,
                "latency": round(time.perf_counter() - t0, 3), "error": None}
    except Exception as exc:
        return {"trigger": None, "missing_items": [], "latency": round(time.perf_counter() - t0, 3),
                "error": type(exc).__name__}


def rule(question: str, have: set[str], patterns: list[tuple[str, list[str]]]) -> bool:
    return any(re.search(pat, question) and not (set(ids) & have) for pat, ids in patterns)


def decoy_item(row: dict) -> str:
    gold = set(row["gold"])
    preferred = row.get("distractors") or [SIMILAR[row["gold"][-1]]]
    return next((item for item in preferred + list(SIMILAR) if item not in gold), "R16")


def score(rows: list[dict], patterns: list[tuple[str, list[str]]]) -> dict:
    counts = {method: {"complete_fp": 0, "complete_n": 0, "missing_hit": 0,
                       "missing_n": 0, "distractor_hit": 0, "distractor_n": 0}
              for method in ("model", "rule", "hybrid")}
    for row in rows:
        gold = set(row["gold"])
        missing = row["gold"][-1]
        complete = gold
        absent = gold - {missing}
        # Deterministic confounder: another responsibility from a similar pair, if supplied.
        decoy = decoy_item(row)
        distractor = absent | {decoy}
        for label, have, should_trigger in (("complete", complete, False), ("missing", absent, True),
                                            ("distractor", distractor, True)):
            r = rule(row["question"], have, patterns)
            m = None if row["predicted"] is None else bool(set(row["predicted"]) - have)
            vals = {"rule": r, "model": m, "hybrid": r if m is None else m}
            for method, value in vals.items():
                if value is None:
                    continue
                if should_trigger:
                    counts[method][f"{label}_n"] += 1
                    counts[method][f"{label}_hit"] += bool(value)
                else:
                    counts[method]["complete_n"] += 1
                    counts[method]["complete_fp"] += bool(value)
        row["rule_complete"] = rule(row["question"], complete, patterns)
        row["rule_missing"] = rule(row["question"], absent, patterns)
        row["model_complete"] = None if row["predicted"] is None else bool(set(row["predicted"]) - complete)
        row["model_missing"] = None if row["predicted"] is None else bool(set(row["predicted"]) - absent)
    return counts


def direct_cases(rows: list[dict]) -> list[tuple[dict, str, set[str]]]:
    cases = []
    for row in rows:
        gold = set(row["gold"])
        absent = gold - {row["gold"][-1]}
        decoy = decoy_item(row)
        cases += [(row, "complete", gold), (row, "missing", absent),
                  (row, "distractor", absent | {decoy})]
    return cases


def latency_probe(client, catalog: str, data: list[dict]) -> dict:
    from agent.pipeline import Pipeline

    pipe = Pipeline(None, client=client)
    samples = data[:10] + [r for r in data if r["source"] == "complex"][:10]
    samples += [r for r in data if r["source"] == "paraphrase"]
    rows = []
    for row in samples:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            understanding = pool.submit(pipe.understand, row["question"], {})
            gate = pool.submit(ask, client, row, catalog)
            understanding.result()
            understand_ready = time.perf_counter() - t0
            gate_result = gate.result()
            both_ready = time.perf_counter() - t0
        rows.append({"id": row["id"], "understand_s": round(understand_ready, 3),
                     "both_s": round(both_ready, 3),
                     "gate_added_s": round(max(0, both_ready - understand_ready), 3),
                     "gate_error": gate_result["error"]})
    return {"n": len(rows), "understand_p50": round(statistics.median(r["understand_s"] for r in rows), 3),
            "both_p50": round(statistics.median(r["both_s"] for r in rows), 3),
            "added_p50": round(statistics.median(r["gate_added_s"] for r in rows), 3),
            "added_max": max(r["gate_added_s"] for r in rows), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--tag", default="v2")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--negative", action="store_true")
    parser.add_argument("--latency-probe", action="store_true")
    args = parser.parse_args()
    if not args.env_file.is_file():
        parser.error("env file not found")
    load_dotenv(args.env_file)
    data = negative_questions() if args.negative else questions()
    patterns = keyword_patterns()
    catalog = "\n".join(f"{item['id']} {item['责任名称']}（{item['责任&加油包']}）" for item in coverage_items())
    client = cloud.llm_client().with_options(timeout=args.timeout, max_retries=0)
    if args.latency_probe:
        summary = {"parallel": latency_probe(client, catalog, data)}
        times = [row["gate_added_s"] for row in summary["parallel"]["rows"]]
        errors = sum(row["gate_error"] is not None for row in summary["parallel"]["rows"])
        data = summary["parallel"]["rows"]
    elif args.negative and args.direct:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(ask_direct, client, row, set(), catalog): row for row in data}
            for future in as_completed(futures):
                row = futures[future]
                row["direct"] = future.result()
        summary = {"negative_direct": {"n": len(data), "false_trigger": 0}}
        for row in data:
            result = row["direct"]
            if result["trigger"] is None:
                result["trigger"] = rule(row["question"], set(), patterns)
                result["fallback"] = True
            summary["negative_direct"]["false_trigger"] += result["trigger"]
        times = sorted(row["direct"]["latency"] for row in data)
        errors = sum(row["direct"]["error"] is not None for row in data)
    elif args.negative:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(ask, client, row, catalog): row for row in data}
            for future in as_completed(futures):
                row = futures[future]
                row.update(future.result())
        summary = {"negative": {"n": len(data), "model_false_trigger": 0, "rule_false_trigger": 0}}
        for row in data:
            m = bool(row["predicted"]) if row["predicted"] is not None else rule(row["question"], set(), patterns)
            row["model_trigger"] = m
            row["rule_trigger"] = rule(row["question"], set(), patterns)
            summary["negative"]["model_false_trigger"] += m
            summary["negative"]["rule_false_trigger"] += row["rule_trigger"]
        times = sorted(row["latency"] for row in data)
        errors = sum(row["error"] is not None for row in data)
    elif args.direct:
        cases = direct_cases(data)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(ask_direct, client, row, have, catalog): (row, label)
                       for row, label, have in cases}
            for future in as_completed(futures):
                row, label = futures[future]
                row[f"direct_{label}"] = future.result()
        summary = {"direct": {"complete_fp": 0, "complete_n": 0, "missing_hit": 0, "missing_n": 0,
                              "distractor_hit": 0, "distractor_n": 0}}
        for row in data:
            for label in ("complete", "missing", "distractor"):
                result = row[f"direct_{label}"]
                if result["trigger"] is None:
                    result["trigger"] = rule(row["question"],
                                              next(have for r, name, have in cases if r is row and name == label),
                                              patterns)
                    result["fallback"] = True
                summary["direct"][f"{label}_{'fp' if label == 'complete' else 'hit'}"] += result["trigger"]
                summary["direct"][f"{label}_n"] += 1
        times = sorted(row[f"direct_{label}"]["latency"] for row in data
                       for label in ("complete", "missing", "distractor"))
        errors = sum(row[f"direct_{label}"]["error"] is not None for row in data
                     for label in ("complete", "missing", "distractor"))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(ask, client, row, catalog): row for row in data}
            for future in as_completed(futures):
                row = futures[future]
                row.update(future.result())
        summary = score(data, patterns)
        times = sorted(row["latency"] for row in data)
        errors = sum(row["error"] is not None for row in data)
    if not args.latency_probe:
        summary["latency"] = {"p50": round(statistics.median(times), 3),
                              "p90": times[int(0.9 * (len(times) - 1))], "max": max(times)}
        summary["n_questions"] = len(data)
    summary["errors"] = errors
    out = OUT_DIR / f"exp16_evidence_gate_{args.tag}.json"
    out.write_text(json.dumps({"summary": summary, "rows": data}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "parallel"} |
                     ({"parallel": {k: v for k, v in summary["parallel"].items() if k != "rows"}}
                      if "parallel" in summary else {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
