"""实验 6（部分）：报价必需条件（槽位）的抽取。

用例在运行前写好，运行后不再针对用例改规则，结果即首次运行结果。
覆盖阿拉伯数字和中文数字年龄、有无社保的多种说法、缴费方式、为谁投保。
意图分类需要接入模型后测试，不在本实验范围。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.speech import extract_slots  # noqa: E402

OUT = ROOT / "experiments" / "results"

CASES: list[tuple[str, dict]] = [
    ("我30岁多少钱", {"age": 30}),
    ("我三十岁多少钱", {"age": 30}),
    ("三十五岁有社保", {"age": 35, "has_social": True}),
    ("我今年42岁，没有社保", {"age": 42, "has_social": False}),
    ("我没社保", {"has_social": False}),
    ("社保断了好几年了", {"has_social": False}),
    ("我一直交社保的", {"has_social": True}),
    ("有医保", {"has_social": True}),
    ("医保没交", {"has_social": False}),
    ("按月交多少", {"pay": "月缴"}),
    ("每个月付的话呢", {"pay": "月缴"}),
    ("一年交一次多少钱", {"pay": "年缴"}),
    ("年缴划算吗", {"pay": "年缴"}),
    ("给我妈买，她六十岁", {"relation": "父母", "age": 60, "gender": "女"}),
    ("给我爸买，他65岁了", {"relation": "父母", "age": 65, "gender": "男"}),
    ("我儿子8岁", {"relation": "子女", "age": 8, "gender": "男"}),
    ("给我女儿买，五岁", {"relation": "子女", "age": 5, "gender": "女"}),
    ("给我老婆买，她28岁有社保", {"relation": "配偶", "age": 28, "has_social": True, "gender": "女"}),
    ("我老公四十五岁", {"relation": "配偶", "age": 45, "gender": "男"}),
    ("我自己买，二十八岁", {"relation": "本人", "age": 28}),
    ("我是男的，三十一岁", {"age": 31, "gender": "男"}),
    ("我女的，40岁，有社保，按月交", {"age": 40, "gender": "女", "has_social": True, "pay": "月缴"}),
    ("不对，我没有社保", {"has_social": False}),
    ("说错了，我是35岁", {"age": 35}),
    ("我父母都七十了还能买吗", {"relation": "父母", "age": 70}),
    ("孩子两岁能买吗", {"relation": "子女", "age": 2}),
    ("我社保在交着", {"has_social": True}),
    ("没买过社保", {"has_social": False}),
    ("我二十九", {"age": 29}),  # 口语常省略"岁"
    ("我妈五十八了", {"relation": "父母", "age": 58, "gender": "女"}),
]


def main() -> dict:
    per_slot: dict[str, list[int]] = {}
    errors = []
    exact = 0
    for text, gold in CASES:
        got = extract_slots(text)
        ok_all = True
        for k, v in gold.items():
            per_slot.setdefault(k, [0, 0])
            per_slot[k][1] += 1
            if got.get(k) == v:
                per_slot[k][0] += 1
            else:
                ok_all = False
        extra_wrong = {k: v for k, v in got.items() if k in ("age", "has_social", "pay") and k not in gold}
        if extra_wrong:
            ok_all = False
        if ok_all:
            exact += 1
        else:
            errors.append({"text": text, "gold": gold, "got": got})
    result = {
        "cases": len(CASES),
        "all_slots_correct": f"{exact}/{len(CASES)}",
        "per_slot_accuracy": {k: f"{c}/{n}" for k, (c, n) in per_slot.items()},
        "errors": errors,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp6_slots.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
