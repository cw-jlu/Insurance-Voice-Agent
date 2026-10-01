"""检索扩充：在第一轮 40 题基础上新增 110 题（共 150 题），覆盖全部 16 项责任、投保须知、责任免除。
新增题在运行前写好，首次运行结果即最终结果。LLM 查询改写需要大模型额度，额度恢复后补测。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agent import cloud  # noqa: E402
from agent.pipeline import HybridRetriever  # noqa: E402
from embed_cache import corpus, load  # noqa: E402
import exp_retrieval as v1  # noqa: E402
import exp_retrieval_v2 as v2  # noqa: E402

OUT = ROOT / "experiments" / "results"
I = lambda r: [("item", r)]  # noqa: E731
D = lambda doc, kw: [("doc", doc, kw)]  # noqa: E731

NEW = [
    ("住院前后看门诊的钱能报吗", I("R01")), ("住院花了三万能报多少", I("R01")), ("特殊门诊费用管不管", I("R01")),
    ("门诊手术能报吗", I("R01")), ("外购药械是什么意思", I("R01")), ("一年最多能报多少住院费", I("R01")),
    ("急诊的救护车费用能报吗", I("R02")), ("门诊检查费用报不报", I("R02")), ("看急诊的药费能报吗", I("R02")),
    ("得了癌症住院怎么报", I("R03")), ("重大疾病有多少种", I("R03")), ("大病门诊能报吗", I("R03")),
    ("脑梗以后的康复费用能报吗", I("R04")), ("康复住院救护车费用有上限吗", I("R04")), ("确诊多久以内的康复能报", I("R04")),
    ("光免疫疗法能报吗", I("R05")), ("质子治疗的床位费有限制吗", I("R05")), ("去日本看病能报吗", I("R05")),
    ("院外买的抗癌药能报吗", I("R06")), ("特药怎么申请", I("R06")), ("海南的进口药能报吗", I("R06")),
    ("社保目录里的药怎么报", I("R06")), ("急需的药能报吗", I("R06")),
    ("转院去别的省坐飞机的钱能报吗", I("R07")), ("陪护家属的住宿费能报吗", I("R07")), ("转诊坐高铁能报一等座吗", I("R07")),
    ("护工一天给多少钱", I("R08")), ("护工最多给几天", I("R08")), ("自己请的护工能报吗", I("R08")),
    ("一家人的免赔额能加在一起算吗", I("R09")), ("家庭共享免赔包括门诊吗", I("R09")),
    ("确诊癌症能直接拿钱吗", I("R10")), ("重疾一次性赔多少", I("R10")),
    ("住国际部能报吗", I("R11")), ("特需病房的床位费有限额吗", I("R11")), ("特需医疗用不用社保都一样吗", I("R11")),
    ("住院花了五千块能报吗", I("R12")), ("看痔疮住院能报多少", I("R12")), ("颈椎病住院能报吗", I("R12")),
    ("感冒去医院门诊能报多少", I("R13")), ("门诊一天去了两次算几次", I("R13")), ("门急诊加油包一年最多报多少", I("R13")),
    ("住院三天有补贴吗", I("R14")), ("住院补贴一年最多几天", I("R14")), ("两次住院间隔很短算一次吗", I("R14")),
    ("重症监护室补贴和住院补贴能一起拿吗", I("R15")), ("ICU每天补多少", I("R15")),
    ("在网上药店买药能报一半吗", I("R16")), ("买药的配送费谁出", I("R16")), ("孩子能用药省保买处方药吗", I("R16")),
    ("买药每个月最多报多少", I("R16")),
    ("谁可以当投保人", D("投保须知", "投保人资格")), ("外国人能买吗", D("投保须知", "固定居所")),
    ("被保险人最大多少岁", D("投保须知", "被保险人年龄")), ("什么职业不能买", D("投保须知", "职业")),
    ("买两份能报两次吗", D("投保须知", "限投保1份")), ("今天买明天能用吗", D("投保须知", "次日零时")),
    ("保障期是多长", D("投保须知", "保险期间")), ("保费会不会涨", D("投保须知", "年龄增长")),
    ("不告诉你们病史会怎样", D("投保须知", "如实告知")), ("总保额是多少", D("投保须知", "总保额")),
    ("有哪些增值服务", D("投保须知", "e享安康")), ("药省保的等待期多久", D("投保须知", "等待期")),
    ("加油包的等待期是多少", D("投保须知", "等待期")), ("投保时要告诉被保险人吗", D("投保须知", "明确说明")),
    ("个人信息怎么保护", D("投保须知", "个人信息")), ("能不能自动扣款", D("投保须知", "代扣")),
    ("得过的病能报吗", D("责任免除", "既往症")), ("投保前就有的病复发了能报吗", D("责任免除", "既往症")),
    ("其他保险报过的还能报吗", D("责任免除", "补偿原则")), ("在诊所看病能报吗", D("责任免除", "二级")),
    ("因为医疗事故产生的费用能报吗", D("责任免除", "医疗事故")), ("性病能报吗", D("责任免除", "性病")),
    ("精神病住院能报吗", D("责任免除", "精神")), ("遗传病能报吗", D("责任免除", "遗传")),
    ("先天畸形能报吗", D("责任免除", "先天")), ("装人工关节能报吗", D("责任免除", "人工关节")),
    ("种牙能报吗", D("责任免除", "义齿")), ("近视手术能报吗", D("责任免除", "屈光不正")),
    ("做胃减容手术能报吗", D("责任免除", "胃减容")), ("矮小症治疗能报吗", D("责任免除", "矮小症")),
    ("流产的费用能报吗", D("责任免除", "流产")), ("试管婴儿能报吗", D("责任免除", "人工受孕")),
    ("做体检能报吗", D("责任免除", "健康体检")), ("买助听器能报吗", D("责任免除", "助听器")),
    ("包皮手术能报吗", D("责任免除", "包皮")), ("捐献器官的费用能报吗", D("责任免除", "捐献")),
    ("保健品能报吗", D("责任免除", "保健")), ("非处方药能报吗", D("责任免除", "非处方药")),
    ("去疗养院能报吗", D("责任免除", "疗养")), ("酒驾出事能赔吗", D("责任免除", "酒")),
    ("打架受伤能赔吗", D("责任免除", "斗殴")), ("自杀能赔吗", D("责任免除", "自杀")),
    ("吸毒导致的病能报吗", D("责任免除", "毒品")), ("战争受伤能赔吗", D("责任免除", "战争")),
    ("核辐射能赔吗", D("责任免除", "核")), ("故意自伤能报吗", D("责任免除", "自伤")),
    ("高风险运动受伤能赔吗", D("责任免除", "潜水|跳伞|攀岩|探险|武术|特技|赛马|赛车")),
    ("无证驾驶出事能赔吗", D("责任免除", "驾驶")), ("被判刑期间生病能报吗", D("责任免除", "羁押|服刑")),
    ("试验性治疗能报吗", D("责任免除", "试验性")), ("怀孕期间生病能报吗", D("责任免除", "怀孕")),
    ("在国外看病能报吗", D("责任免除", "境外|中华人民共和国境外")),
    ("艾滋病能报吗", D("责任免除", "艾滋")), ("整形手术能报吗", D("责任免除", "整形")),
    ("戒烟戒酒治疗能报吗", D("责任免除", "戒")), ("睡眠障碍能报吗", D("责任免除", "睡眠")),
]


def hit(c, conds):
    import re
    for cond in conds:
        if cond[0] == "item" and c.meta.get("item") == cond[1]:
            return True
        if cond[0] == "doc" and c.doc == cond[1] and re.search(cond[2], c.text):
            return True
    return False


def main():
    chunks = corpus(False)
    # make sure every new query's gold actually exists in the corpus; drop and report the ones that don't
    valid, invalid = [], []
    for q, conds in NEW:
        (valid if any(hit(c, conds) for c in chunks) else invalid).append((q, conds))
    v1.QUERIES = v1.QUERIES + valid
    v2.QUERIES = v1.QUERIES
    v2.hit = hit
    r = HybridRetriever(chunks, load(chunks, "clauses"), client=cloud.llm_client())
    res = {"queries": len(v1.QUERIES), "new_queries_valid": len(valid), "new_queries_no_gold_in_corpus": [q for q, _ in invalid],
           "methods": {"tfidf": v2.evaluate(r, "lexical", None), "dense": v2.evaluate(r, "dense", None),
                       "tfidf+dense": v2.evaluate(r, "hybrid", None)}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp_retrieval_v3.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    res = main()
    print(res["queries"], "queries; dropped (no gold in corpus):", res["new_queries_no_gold_in_corpus"])
    for m, v in res["methods"].items():
        print(m, {k: x for k, x in v.items() if k != "misses@5"}, "misses@5:", len(v["misses@5"]))
