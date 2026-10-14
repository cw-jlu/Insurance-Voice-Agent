"""Deterministic premium lookup used by the explicitly scoped legacy sales step.

The evidence consultation path separately reasons from cited rate rows; this
engine does not implement all underwriting, renewal or household-order rules.

Eligibility comes from 投保须知 (被保人 30天–70周岁; 门急诊加油包 ≤55周岁), not from whether
the rate table happens to have a price: the tables also list renewal prices up to 105.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.knowledge import rate_table

MIN_AGE_DAYS = 30
MAX_AGE_NEW = 70  # 投保须知 二.2
ADDON_MAX_AGE = {"门急诊医疗加油包": 55}  # 投保须知 二.2

# Family-order discount by headcount (费率表 文字层 (二)1)).
FAMILY_DISCOUNT = {1: 0.0, 2: 0.05, 3: 0.10, 4: 0.15}
FAMILY_DISCOUNT_5_PLUS = 0.20

ADDONS = [
    "家庭共享免赔额加油包",
    "重疾关爱加油包",
    "重大疾病扩展特需医疗加油包",
    "住院医疗费用补偿加油包",
    "门急诊医疗加油包",
    "住院津贴加油包",
    "药省保加油包",
]


@dataclass
class Insured:
    age: int  # completed years; 0 means under 1 year
    has_social: bool
    gender: str | None = None  # "男" / "女", needed only for 重疾关爱
    age_days: int | None = None  # only meaningful when age == 0
    addons: list[str] = field(default_factory=list)
    critical_units: int = 1  # 重疾关爱: number of 5万 units


@dataclass
class Line:
    label: str
    amount: float
    source: str


@dataclass
class QuoteResult:
    ok: bool
    lines: list[Line] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)  # missing slots to ask for
    discount_rate: float = 0.0
    total: float = 0.0
    pending_confirmation: list[str] = field(default_factory=list)
    unit: str = ""

    def numbers(self) -> set[float]:
        """Every number the speech layer is allowed to say for this quote: amounts, total,
        age-band bounds (e.g. "26到30岁"), and the discount percentage."""
        out = {round(l.amount, 2) for l in self.lines}
        out.add(round(self.total, 2))
        for l in self.lines:
            out |= {float(x) for x in re.findall(r"\[(\d+), (\d+)\]", l.source)[0]}
        if self.discount_rate:
            out.add(round(self.discount_rate * 100, 2))
        return out


def family_discount(headcount: int) -> float:
    if headcount >= 5:
        return FAMILY_DISCOUNT_5_PLUS
    return FAMILY_DISCOUNT.get(headcount, 0.0)


def _addon_column(addon: str, insured: Insured) -> str | None:
    if addon == "家庭共享免赔额加油包":
        return f"{addon}_{'有社保' if insured.has_social else '无社保'}"
    if addon == "重疾关爱加油包":
        if insured.gender not in ("男", "女"):
            return None
        return f"{addon}_每5万保额_{insured.gender}"
    return f"{addon}_有无社保"


def quote(insured: list[Insured] | Insured, pay: str, family_size: int | None = None) -> QuoteResult:
    """Quote one order. ``pay`` is "年缴" or "月缴". ``family_size`` counts the household members
    already insured in the series plus this order (defaults to the number insured here)."""
    people = insured if isinstance(insured, list) else [insured]
    res = QuoteResult(ok=True)
    if pay not in ("年缴", "月缴"):
        res.ok = False
        res.needs.append("缴费方式")
        return res
    base_tbl = rate_table(f"{pay}_必选")
    addon_tbl = rate_table(f"{pay}_加油包")
    unit = "元/年" if pay == "年缴" else "元/月"

    for idx, p in enumerate(people, 1):
        who = f"被保人{idx}"
        if p.age == 0 and p.age_days is not None and p.age_days < MIN_AGE_DAYS:
            res.errors.append(f"{who}出生未满30天，不能投保（投保须知 二.2）")
            continue
        if p.age > MAX_AGE_NEW:
            res.errors.append(f"{who}（{p.age}周岁），超过首次投保年龄上限70周岁（投保须知 二.2）")
            continue
        col = "必选计划_有社保" if p.has_social else "必选计划_无社保"
        price, band = base_tbl.price(p.age, col)
        if price is None:
            res.errors.append(f"{who}费率表无对应价格")
            continue
        res.lines.append(Line(f"{who} 必选计划", price, f"{pay}_必选.csv {list(band)} {col}"))

        for addon in p.addons:
            limit = ADDON_MAX_AGE.get(addon, MAX_AGE_NEW)
            if p.age > limit:
                res.errors.append(f"{who}（{p.age}周岁），{addon}投保年龄上限{limit}周岁（投保须知 二.2）")
                continue
            acol = _addon_column(addon, p)
            if acol is None:
                res.needs.append(f"{who}性别")
                continue
            aprice, aband = addon_tbl.price(p.age, acol)
            if aprice is None:
                res.errors.append(f"{who}{addon}在该年龄段不可选")
                continue
            if addon == "重疾关爱加油包":
                aprice *= p.critical_units
            res.lines.append(Line(f"{who} {addon}", aprice, f"{pay}_加油包.csv {list(aband)} {acol}"))

    headcount = family_size if family_size is not None else len(people)
    res.discount_rate = family_discount(headcount)
    subtotal = sum(l.amount for l in res.lines)
    if res.discount_rate:
        # 数据只说"当前订单每人价格优享"，是否覆盖加油包、如何取整未说明 → 标记待确认。
        res.pending_confirmation.append("家庭单折扣是否作用于加油包、折后取整规则")
        res.total = round(subtotal * (1 - res.discount_rate), 2)
    else:
        res.total = round(subtotal, 2)
    if res.errors or res.needs:
        res.ok = False
    res.unit = unit
    return res
