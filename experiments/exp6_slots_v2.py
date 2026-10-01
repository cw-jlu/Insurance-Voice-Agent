"""实验 6 扩充：条件抽取，约 200 句，由模板组合生成（固定随机种子），首次运行结果即最终结果。

模板中特意包含第一轮没见过的说法：不带"岁"的年龄、"周岁"、"今年""快"等修饰、社保的多种否定说法。
这里测的是规则抽取；方案中由 LLM 抽取、规则兜底，LLM 抽取待模型额度恢复后测试。
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.speech import extract_slots  # noqa: E402

OUT = ROOT / "experiments" / "results"
CN = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}


def cn_num(n: int) -> str:
    if n <= 10:
        return CN[n] if n != 2 else "两"
    t, o = divmod(n, 10)
    return (("" if t == 1 else CN[t].replace("两", "二")) + "十" + (CN[o].replace("两", "二") if o else ""))


AGE_FORMS = [
    ("{n}岁", "digit+岁"), ("{c}岁", "中文+岁"), ("{n}周岁", "digit+周岁"), ("今年{n}", "今年+digit"),
    ("{c}了", "中文+了"), ("{n}了", "digit+了"), ("快{n}了", "快+digit"), ("{c}", "中文裸数"),
]
WHO = [("我", "本人", None), ("我自己", "本人", None), ("我妈", "父母", "女"), ("我爸", "父母", "男"),
       ("我老婆", "配偶", "女"), ("我老公", "配偶", "男"), ("我儿子", "子女", "男"), ("我女儿", "子女", "女"),
       ("我父亲", "父母", "男"), ("我母亲", "父母", "女"), ("我爱人", "配偶", None), ("孩子", "子女", None)]
SOCIAL = [("有社保", True), ("有医保", True), ("社保一直在交", True), ("单位给交社保", True), ("有城乡医保", True),
          ("没有社保", False), ("没社保", False), ("社保断了", False), ("没交医保", False), ("没买过社保", False),
          ("社保停了", False), ("医保早就不交了", False)]
PAY = [("按月交", "月缴"), ("每个月付", "月缴"), ("月缴", "月缴"), ("一年交一次", "年缴"), ("按年交", "年缴"), ("年缴", "年缴")]


def build(n_cases: int = 200) -> list[tuple[str, dict, str]]:
    rng = random.Random(20261004)
    cases = []
    for _ in range(n_cases):
        who, rel, gender = rng.choice(WHO)
        age = rng.randint(1, 70) if rel != "子女" else rng.randint(1, 25)
        if rel == "父母":
            age = rng.randint(45, 70)
        form, form_name = rng.choice(AGE_FORMS)
        age_txt = form.format(n=age, c=cn_num(age))
        parts, gold = [f"{who}{age_txt}"], {"age": age, "relation": rel}
        if gender:
            gold["gender"] = gender
        if rng.random() < 0.7:
            s_txt, s_val = rng.choice(SOCIAL)
            parts.append(s_txt)
            gold["has_social"] = s_val
        if rng.random() < 0.4:
            p_txt, p_val = rng.choice(PAY)
            parts.append(p_txt)
            gold["pay"] = p_val
        parts.append(rng.choice(["多少钱", "一年多少", "能买吗", "怎么买", "价格是多少"]))
        cases.append(("，".join(parts), gold, form_name))
    return cases


def main() -> dict:
    cases = build()
    per_slot: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    per_form: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    exact, errors = 0, []
    for text, gold, form in cases:
        got = extract_slots(text)
        ok = True
        for k, v in gold.items():
            per_slot[k][1] += 1
            if got.get(k) == v:
                per_slot[k][0] += 1
            else:
                ok = False
        per_form[form][1] += 1
        per_form[form][0] += int(got.get("age") == gold["age"])
        exact += ok
        if not ok and len(errors) < 25:
            errors.append({"text": text, "gold": gold, "got": got})
    result = {
        "cases": len(cases),
        "all_slots_correct": f"{exact}/{len(cases)}",
        "per_slot_accuracy": {k: f"{c}/{n} ({c / n:.1%})" for k, (c, n) in per_slot.items()},
        "age_accuracy_by_form": {k: f"{c}/{n} ({c / n:.1%})" for k, (c, n) in per_form.items()},
        "error_examples": errors,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp6_slots_v2.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != "error_examples"}, ensure_ascii=False, indent=2))
    for e in r["error_examples"][:12]:
        print(e)
