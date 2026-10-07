"""第八轮验收后修复的独立复核集（验收后修复完成后、首次运行前写好；全部为保留集）。

qa_v4 的错题用于了验收后的修复，qa_v4 因此也变成开发集。本集 40 题与 qa_v2/v3/v4 不重复，
专门包含验收后修复涉及的题型的新说法（问保障时带"多少"、年龄资格、身份误判、多轮补条件），用来检验修复是否泛化。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from build_qa_v3 import CONFIRM, NOT, UW, M, Y, nre, price  # noqa: E402

OUT = ROOT / "experiments" / "data" / "qa_v5.json"
items = []
ADDON = r"加油包|另外|额外|加购|单独"


def add(cat, q, must=(), must_not=(), facts="", context=None, turns=None):
    items.append({"category": cat, "question": q, "must": list(must), "must_not": list(must_not), "facts": facts,
                  "context": context or {}, "turns": turns, "source": "v5", "split": "heldout", "valid": True})


PRICE_NOT = r"基础版一年\d+元"  # a coverage question must not be answered with the base price
for rid, q, must, f in [
    ("R01", "住院的钱一年最多能报多少", [r"300\s*万"], "一般医疗300万"),
    ("R03", "得了尿毒症透析，免赔额要扣多少", [r"(没有|无|不设|0).{0,6}免赔|免赔.{0,4}(0|没有)"], "重大疾病医疗无免赔额"),
    ("R05", "质子治疗一年最高能赔多少钱", [r"600\s*万"], "恶性肿瘤先进疗法600万"),
    ("R08", "大病住院护工费一天报多少", [r"500\s*元|五百"], "重疾护工500元/天"),
    ("R10", "确诊癌症能拿多少钱的一次性赔款", [r"5\s*万", ADDON], "重疾关爱加油包5万，需另外选购"),
    ("R13", "看门诊的加油包一次最多赔多少", [r"300\s*元|三百"], "门急诊加油包次限额300元"),
    ("R14", "普通住院每天的津贴是多少", [r"100\s*元|一百", ADDON], "住院津贴100元/天，需加油包"),
    ("R15", "重症监护每天补贴多少钱", [r"500\s*元|五百", ADDON], "ICU津贴500元/天，需加油包"),
    ("R16", "网上药店买药一年最多报多少", [r"1000\s*元|一千|1千"], "药省保年1000元"),
    ("R07", "转院去外地看大病，路费和住宿一年报多少", [r"1\s*万|一万"], "重疾异地转诊交通住宿年1万"),
    ("R11", "特需部的床位费一天能报多少", [r"1500\s*元|一千五", ADDON], "特需医疗加油包床位费1500元/天"),
    ("R04", "脑梗后在康复医院住院，报销比例多少", [r"80\s*%|60\s*%"], "特定疾病康复住院80%/60%"),
]:
    add("保障", q, must, [PRICE_NOT], f"{rid} {f}")

rng = random.Random(20261120)
for _ in range(8):
    age = rng.choice([4, 8, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 69])
    social = rng.random() < 0.5
    pay = "月缴" if rng.random() < 0.4 else "年缴"
    who = rng.choice(["我", "我爱人", "我家老大"]) if age >= 18 else "我家小孩"
    s_txt = rng.choice(["有社保", "有居民医保"]) if social else rng.choice(["没社保", "医保没交"])
    p_txt = "按月得交多少" if pay == "月缴" else "一年保费多少"
    v = price(age, social, M if pay == "月缴" else Y)
    add("价格", f"{who}{age}岁，{s_txt}，{p_txt}", [nre(v)], [nre(price(age, not social, M if pay == "月缴" else Y))],
        f"{age}岁{'有' if social else '无'}社保{pay}{v:g}元")

for q in ["做近视激光手术能报吗", "镶假牙的钱能报吗", "体检的费用报不报", "整容失败修复能报吗", "产检能报吗"]:
    add("免责", q, [NOT], [r"(?<!不)(?<!不能)(可以|能)报[。！]"], "责任免除")

for q in ["我有痛风还吃着药，能投保吗", "我妈做过心脏搭桥，能给她买吗", "我有肾结石，买了以后结石住院能报吗"]:
    add("缺口", q, [UW], [r"(?<!不)(?<!不能)(?<!能不)(可以|能|一般能)(买|投保|承保)[。！]|没问题|不影响投保"], "以核保为准")
for q in ["这保险能不能一直续到八十岁", "到期以后是不是还得重新审核才能续"]:
    add("缺口", q, [CONFIRM], [r"(?<!不)(?<!不能)(?<!无法)保证续保|一定(能)?续保|(可以|能)一直(续|保)"], "无续保规则")
add("缺口", "医保报掉的部分能抵那一万块的免赔吗", [CONFIRM], [r"(?<!不)(?<!能不)(?<!能否)(可以|能|会)(算进|计入|抵扣|算在|抵)"], "无此规则")

for q, must, mn, f, cat in [
    ("我今年69岁，还能买这个吗", [r"(可以|能)(买|投保)|70\s*(周)?岁"], [r"不能(买|投保)[。！]"], "69岁可首次投保（≤70）", "投保条件"),
    ("我58了，门诊加油包能买吗", [r"55\s*(周)?岁"], [r"(?<!不)(可以|能)买"], "门急诊加油包限55周岁及以下", "投保条件"),
    ("你们这个电话是人工打的吗", [r"AI|人工智能|智能助理"], [r"我是真人"], "必须承认是AI", "身份"),
    ("没有，我身体一直挺好的", [], [r"我是.{0,6}(AI|人工智能|智能助理)"], "不是身份询问，不应回答身份", "身份"),
]:
    add(cat, q, must, mn, f)

for t1, t2, must, mn, f in [
    ("我爱人今年44", "她是有医保的，保费一年多少", [nre(price(44, True, Y))], [], f"44岁有社保年缴{price(44, True, Y):g}"),
    ("我31岁有社保", "说错了，社保去年就断了", [nre(price(31, False, Y))], [nre(price(31, True, Y))], f"31岁无社保{price(31, False, Y):g}"),
    ("我36岁有社保", "那ICU一天补多少", [r"500\s*元|五百", ADDON], [PRICE_NOT], "ICU津贴500元/天"),
    ("给我爸问的，他66岁有社保", "他这个岁数买有什么限制吗", [r"70\s*(周)?岁"], [], "首次投保≤70周岁"),
]:
    add("多轮", t2, must, mn, f, turns=[t1, t2])


def main():
    for i, it in enumerate(items):
        it["id"] = f"W{i + 1:03d}"
    old = set()
    for p in ("qa_v2.json", "qa_v3.json", "qa_v4.json"):
        old |= {x["question"] for x in json.loads((ROOT / "experiments" / "data" / p).read_text(encoding="utf-8"))}
    dup = [i["question"] for i in items if i["question"] in old]
    assert not dup, dup
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    import collections
    print(len(items), collections.Counter(i["category"] for i in items))


if __name__ == "__main__":
    main()
