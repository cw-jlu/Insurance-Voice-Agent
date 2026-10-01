"""实验 1：报价全量枚举。

用两套独立实现比对：被测的报价引擎（src/agent/quote.py）和这里用 pandas 写的参照实现。
同时检查年龄资格、边界年龄、不可选加油包、家庭单折扣、月缴与年缴关系。
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.quote import ADDONS, Insured, family_discount, quote  # noqa: E402

PREMIUM = ROOT / "data" / "clean" / "premium"
OUT = ROOT / "experiments" / "results"


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(PREMIUM / f"{name}.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    bounds = df["年龄"].str.strip("[]").str.split(",", expand=True).astype(int)
    df["lo"], df["hi"] = bounds[0], bounds[1]
    return df


TABLES = {n: load(n) for n in ("年缴_必选", "月缴_必选", "年缴_加油包", "月缴_加油包")}


def oracle(age: int, social: bool, pay: str, gender: str, addon: str | None) -> float | None:
    """Reference price for one person, or None if not purchasable."""
    if age > 70 or (addon == "门急诊医疗加油包" and age > 55):
        return None
    base = TABLES[f"{pay}_必选"]
    row = base[(base.lo <= age) & (base.hi >= age)].iloc[0]
    total = float(row["必选计划_有社保" if social else "必选计划_无社保"])
    if addon:
        t = TABLES[f"{pay}_加油包"]
        r = t[(t.lo <= age) & (t.hi >= age)].iloc[0]
        if addon == "家庭共享免赔额加油包":
            col = f"{addon}_{'有社保' if social else '无社保'}"
        elif addon == "重疾关爱加油包":
            col = f"{addon}_每5万保额_{gender}"
        else:
            col = f"{addon}_有无社保"
        v = r[col].strip()
        if v in ("", "-"):
            return None
        total += float(v)
    return total


def main() -> dict:
    cases = mismatches = 0
    rejected_ok = rejected_wrong = 0
    examples: list[dict] = []
    for age, social, pay, gender, addon in itertools.product(
        range(0, 106), (True, False), ("年缴", "月缴"), ("男", "女"), [None, *ADDONS]
    ):
        cases += 1
        expected = oracle(age, social, pay, gender, addon)
        r = quote(Insured(age=age, has_social=social, gender=gender, addons=[addon] if addon else []), pay)
        got = r.total if r.ok else None
        if expected is None:
            if got is None:
                rejected_ok += 1
            else:
                rejected_wrong += 1
                examples.append({"age": age, "addon": addon, "pay": pay, "expected": None, "got": got})
        elif got is None or abs(got - expected) > 1e-6:
            mismatches += 1
            examples.append({"age": age, "social": social, "pay": pay, "gender": gender, "addon": addon,
                             "expected": expected, "got": got, "errors": r.errors})

    # Boundary spot checks against values stated in the solution document.
    spot = {
        "30岁有社保年缴": (quote(Insured(30, True), "年缴").total, 318.0),
        "30岁无社保年缴": (quote(Insured(30, False), "年缴").total, 679.0),
        "30岁有社保月缴": (quote(Insured(30, True), "月缴").total, 31.0),
        "30岁无社保月缴": (quote(Insured(30, False), "月缴").total, 67.0),
        "31岁有社保年缴": (quote(Insured(31, True), "年缴").total, 423.0),
        "70岁可投保": (quote(Insured(70, True), "年缴").ok, True),
        "71岁不可首次投保": (quote(Insured(71, True), "年缴").ok, False),
        "出生20天不可投保": (quote(Insured(0, True, age_days=20), "年缴").ok, False),
        "55岁可选门急诊加油包": (quote(Insured(55, True, addons=["门急诊医疗加油包"]), "年缴").ok, True),
        "56岁不可选门急诊加油包": (quote(Insured(56, True, addons=["门急诊医疗加油包"]), "年缴").ok, False),
        "重疾关爱缺性别需追问": (quote(Insured(40, True, addons=["重疾关爱加油包"]), "年缴").needs, ["被保人1性别"]),
    }
    spot_fail = {k: v for k, v in spot.items() if v[0] != v[1]}

    # Family discount: 1-6 people, base plan only.
    family = {}
    for n in range(1, 7):
        people = [Insured(30, True) for _ in range(n)]
        r = quote(people, "年缴")
        family[n] = {"rate": r.discount_rate, "total": r.total, "expected": round(318 * n * (1 - family_discount(n)), 2),
                     "pending": bool(r.pending_confirmation)}
    family_fail = {n: v for n, v in family.items() if abs(v["total"] - v["expected"]) > 1e-6}
    expected_rates = {1: 0, 2: 0.05, 3: 0.10, 4: 0.15, 5: 0.20, 6: 0.20}
    rate_fail = {n: family[n]["rate"] for n in family if family[n]["rate"] != expected_rates[n]}

    # Monthly × 12 vs annual for the base plan, eligible ages only.
    mvy = []
    for (_, a), (_, m) in zip(TABLES["年缴_必选"].iterrows(), TABLES["月缴_必选"].iterrows()):
        if a["lo"] > 70:
            continue
        for col in ("必选计划_有社保", "必选计划_无社保"):
            mvy.append({"band": a["年龄"], "col": col, "annual": float(a[col]), "monthly_x12": float(m[col]) * 12})
    monthly_more = sum(1 for x in mvy if x["monthly_x12"] > x["annual"])

    # Renewal-only prices that exist in the tables but must NOT be quoted to new buyers.
    addon_tbl = TABLES["年缴_加油包"]
    outpatient_renewal_only = addon_tbl[(addon_tbl.lo > 55) & (addon_tbl.lo <= 70)]["门急诊医疗加油包_有无社保"].tolist()
    base_renewal_bands = int((TABLES["年缴_必选"].lo > 70).sum())

    result = {
        "cases": cases,
        "price_mismatches": mismatches,
        "ineligible_correctly_rejected": rejected_ok,
        "ineligible_wrongly_quoted": rejected_wrong,
        "accuracy": round(1 - (mismatches + rejected_wrong) / cases, 6),
        "mismatch_examples": examples[:10],
        "spot_checks": {k: {"got": v[0], "expected": v[1]} for k, v in spot.items()},
        "spot_failures": list(spot_fail),
        "family_discount": family,
        "family_failures": list(family_fail) + [f"rate@{n}" for n in rate_fail],
        "monthly_x12_more_than_annual": f"{monthly_more}/{len(mvy)}",
        "renewal_only_prices": {
            "必选计划_年龄段>70": base_renewal_bands,
            "门急诊加油包_56-70岁有价格但首次投保不可选": outpatient_renewal_only,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp1_quote.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k not in ("family_discount", "spot_checks")}, ensure_ascii=False, indent=2))
