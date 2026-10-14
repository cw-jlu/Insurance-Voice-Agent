"""Build and check operator-owned dialogue cases; never calls a cloud model.

The JSON is a scenario specification, not input for the old single-answer runner.
Hidden facts, replies and oracles must stay outside the agent's prompt/profile.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/data/complex_dialogue_v1.json"
DOC = ROOT / "docs/25-复杂问题测试集与用例.md"
YEAR = "data/clean/premium/年缴_必选.csv"
MONTH = "data/clean/premium/月缴_必选.csv"
NOTICE = "data/clean/text/尊享E生百万医疗2026投保须知.md"
EXCLUSION = "data/clean/text/尊享E生百万医疗2026责任免除.md"
COVERAGE = "data/clean/保险责任.md"
PRICE_NOTE = "data/clean/text/尊享E生百万医疗2026年缴费率.md"
CASES: list[dict] = []
SOURCES: dict[str, dict] = {}


def source(key: str, path: str, excerpt: str) -> str:
    raw = (ROOT / path).read_bytes()
    lines = raw.decode("utf-8-sig").splitlines()
    matches = [i for i, line in enumerate(lines, 1) if excerpt in line]
    if not matches:
        raise ValueError(f"Source excerpt missing: {key}")
    SOURCES[key] = {"path": path, "line": matches[0], "excerpt": excerpt,
                    "sha256": hashlib.sha256(raw).hexdigest()}
    return key


def setup_sources() -> None:
    source("age", NOTICE, "被保险人年龄：投保时年龄为出生满30天至70周岁（含）")
    source("occupation", NOTICE, "被保险人不属于《众安保险特殊职业类别表》中的职业类别")
    source("underwriting", NOTICE, "若核保通过")
    source("disclosure", NOTICE, "未如实告知")
    source("first_age", PRICE_NOTE, "*首次投保年龄为出生满30天-70周岁")
    source("family2", PRICE_NOTE, "如累计为2人，当前订单每人价格优享5%")
    source("family3", PRICE_NOTE, "当前订单每人价格优享10%")
    source("family4", PRICE_NOTE, "如累计为4人，当前订单每人价格优享15%")
    source("family5", PRICE_NOTE, "如累计为5人及以上，当前订单每人价格优享20%")
    source("family_scope", PRICE_NOTE, "同一投保人名下尊享医疗险系列+众民保医疗险系列产品保单家庭成员")
    source("deductible", COVERAGE, "`D8`")
    source("no_social_settlement", COVERAGE, "`D9`")
    source("shared_basic", COVERAGE, "`D10`")
    source("waiting", COVERAGE, "`D5`")
    source("compensation", EXCLUSION, "已从医保等其他任何途径获得医疗费报销或补偿的部分，不予理赔")
    source("preexisting", EXCLUSION, "除另有约定外，因既往症")
    source("family_deductible", COVERAGE, "`D98`")
    source("general_allowance", COVERAGE, "`D116`")
    source("icu_allowance", COVERAGE, "`D118`")
    source("drug_waiting", COVERAGE, "`D120`")
    source("drug_ratio", COVERAGE, "`D121`")
    source("drug_limit", COVERAGE, "`D122`")
    source("drug_prescription", COVERAGE, "`D123`")
    source("manifest", "data/README.md", "健康告知")
    source("missing_attachments", "data/README.md", "100 种重大疾病清单、特药清单")


def reply(fields: list[str], text: str, reveals: dict,
          check: str, mode: str = "when_requested") -> dict:
    return {"mode": mode, "trigger": "追问下列任一未披露条件时提供；可合并回答，但不提前透露其他条件"
            if mode == "when_requested" else
            "在上一轮正式回答之后主动更正" if mode == "after_answer" else
            "仅当坐席确实追问相关事项时提供；有依据的提前安全转人工也可结束该分支",
            "requested_fields": fields, "utterance": text, "reveals": reveals,
            "checkpoint": check}


def obligation(key: str, text: str, refs: list[str]) -> dict:
    return {"id": key, "expected": text, "source_ids": refs}


def quote(age: int, social: bool, mode: str = "year", after: str = "all_replies") -> dict:
    path = YEAR if mode == "year" else MONTH
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    for line, row in enumerate(rows[1:], 2):
        lo, hi = json.loads(row[0])
        if lo <= age <= hi:
            key = f"rate_{mode}_{lo}_{hi}_{'social' if social else 'no_social'}"
            source(key, path, ','.join(row[1:]))
            # Use the row's actual line, even if identical values occur elsewhere.
            SOURCES[key]["line"] = line
            return {"kind": "standard_premium", "after": after,
                    "inputs": {"age": age, "has_social": social, "first_insurance": True,
                               "plan": "必选计划", "payment": mode, "policy_count": 1},
                    "expected": float(Decimal(row[1 if social else 2])),
                    "unit": "元/年" if mode == "year" else "元/月", "source_ids": [key, "first_age"]}
    raise ValueError(f"Age outside table: {age}")


def difference(age: int, social: bool, after: str = "all_replies") -> dict:
    y, m = quote(age, social), quote(age, social, "month")
    return {"kind": "annual_payment_difference", "after": after,
            "inputs": {"annual": y["expected"], "monthly": m["expected"], "months": 12},
            "expected": float(Decimal(str(m["expected"])) * 12 - Decimal(str(y["expected"]))),
            "unit": "元", "formula": "monthly * 12 - annual",
            "source_ids": y["source_ids"] + m["source_ids"]}


def claim(net: int, deductible: int, ratio: str) -> dict:
    return {"kind": "illustrative_claim", "after": "all_replies",
            "inputs": {"eligible_net_expense": net, "remaining_deductible": deductible,
                       "ratio": ratio, "all_other_contract_conditions_met": True,
                       "no_other_compensation": True},
            "expected": float(max(Decimal(net - deductible), Decimal(0)) * Decimal(ratio)),
            "unit": "元", "formula": "max(eligible_net_expense - remaining_deductible, 0) * ratio",
            "source_ids": ["deductible", "no_social_settlement", "compensation"],
            "note": "仅为用户明确假设下的测算，不代表真实理赔审核结果。"}


def add(prefix: str, title: str, initial: str, hidden: dict, replies: list[dict],
        expected: list[dict], checks: list[str], oracles: list[dict] | None = None) -> None:
    n = sum(c["id"].startswith(prefix) for c in CASES) + 1
    CASES.append({"id": f"{prefix}{n:02d}", "title": title,
                  "type": {"MT": "multi_turn", "CP": "compound", "MX": "mixed"}[prefix],
                  "split": "dev" if n <= 6 else "evaluation_candidate",
                  "initial_user": initial, "initial_profile": {}, "initial_history": [],
                  "hidden_user_facts": hidden, "user_reply_rules": replies,
                  "obligations": expected, "conversation_checks": checks,
                  "numeric_oracles": oracles or [], "status": "not_run"})


def build_cases() -> None:
    q = lambda text="在已披露条件下给必选计划个人单标准保费，标明年/月口径，不将报价当作核保通过": obligation("price", text, ["first_age"])
    eligibility = lambda: obligation("eligibility", "说明年龄是否符合首次投保范围；实际承保还须符合职业、健康告知和核保等条件，不保证通过", ["age", "occupation", "underwriting"])
    discount = lambda n: obligation("discount", f"符合家庭单统计条件时，累计{n}人对应优惠{ {2:5,3:10,4:15,5:20}[n] }%；说明同一投保人和适用产品/家庭成员口径，不无条件承诺个人单打折", [f"family{n}", "family_scope"])
    wait = lambda: obligation("waiting", "区分疾病30天等待期与意外伤害不受该疾病等待期限制，不能把生效等同全部责任立即可赔", ["waiting"])
    handoff = lambda key="underwriting": obligation(key, "现有材料没有疾病核保规则/健康告知问卷，不能确认个体承保；如实告知，并明确转人工核实事项和已有条件", ["manifest", "disclosure", "underwriting"])
    scope = "等待期、医院、费用范围和其他责任条件均满足，无其他补偿，额度充足，非既往症，不加任何加油包"

    add("MT", "分两次补年龄和医保", "我给自己第一次买必选计划，个人单年缴多少钱？",
        {"age":35,"has_social":True}, [reply(["age"],"35岁。",{"age":35},"仍缺医保，继续确认，不能猜价格"),
        reply(["has_social"],"有职工医保。",{"has_social":True},"找回年缴原问项，给423元")], [q()],
        ["缺年龄或医保时先追问；可说明条件分支，但不能替用户选一档", "答复短句35岁时不得误当新问题，不能重复问已确认年龄"], [quote(35,True)])
    add("MT", "只缺医保", "我40岁，给自己首次买必选计划，个人单年缴多少钱？", {"has_social":False},
        [reply(["has_social"],"没有社保。",{"has_social":False},"按40岁无社保报价1330元")], [q()],
        ["不重复问40岁，不默认有社保"], [quote(40,False)])
    add("MT", "孩子月缴补条件", "我想给女儿首次买必选计划，个人单按月交多少钱？", {"age":6,"has_social":True},
        [reply(["age"],"她6周岁。",{"age":6},"仍保留月缴意图"),reply(["has_social"],"她有居民医保。",{"has_social":True},"报价27元/月，不能报成年人的费率")],
        [q()], ["父母身份不替代孩子年龄；合并问年龄和医保时可合并两个回复"], [quote(6,True,"month")])
    add("MT", "家庭优惠先确认统计范围", "家庭一起买能优惠多少？", {"family_count":3,"qualified_family":True},
        [reply(["family_count","qualified_family"],"我是同一个投保人，自己、配偶和女儿共3人，都是首次买你说的适用系列，这次同一订单，没有其他旧保单人数。",{"family_count":3,"qualified_family":True},"给10%并说明统计口径")], [discount(3)],
        ["信息不足时可以先介绍2/3/4/5人档位，不能擅自按三人给实际折扣"])
    add("MT", "理赔测算补身份和剩余免赔", f"我一般住院合规费用15000元，没走医保，{scope}。能测算赔多少吗？", {"has_social":True,"remaining_deductible":4000},
        [reply(["has_social"],"投保选的是有医保身份。",{"has_social":True},"不能将没结算医保等同无医保投保"),
         reply(["remaining_deductible"],"年度免赔只剩4000元。",{"remaining_deductible":4000},"(15000-4000)*60%=6600元")],
        [obligation("calculation","在给定假设下测算6600元，明确60%适用前提及测算性质",["deductible","no_social_settlement"])],
        ["不能默认免赔仍剩10000元；已给合规费用不要求重说", "允许先列完整条件分支，最终必须落到实际补充条件"], [claim(15000,4000,"0.6")])
    add("MT", "免赔已用完的续问", f"一般住院总费用20000元，医保报8000元，剩下12000元全是合规费用，按有医保身份投保并结算，{scope}。还能赔多少？", {"remaining_deductible":0},
        [reply(["remaining_deductible"],"今年的免赔已经全部用完了。",{"remaining_deductible":0},"回答12000元，不再扣一万")],
        [obligation("calculation","剩余免赔为0时测算12000元，不重复计算医保已补偿部分",["deductible","compensation"])],
        ["不能把用户说的医保8000元直接算成免赔使用额", "补条件后仍记得20000/8000/12000"], [claim(12000,0,"1")])
    add("MT", "跨费率档更正年龄", "我35岁有社保，给自己首次买必选计划，个人单年缴多少钱？", {"corrected_age":36},
        [reply([],"我刚说错了，是36岁，其他条件不变，重新报一下。",{"age":36},"按36岁更新517元", "after_answer")], [q()],
        ["两轮都检查；更正后不得沿用423元或只确认年龄不重新报价"], [quote(35,True,after="initial_answer"),quote(36,True)])
    add("MT", "医保更正重算", "我40岁有社保，首次买必选计划，个人单年缴多少钱？", {"corrected_has_social":False},
        [reply([],"更正一下，我没有社保，重新报多少钱？",{"has_social":False},"保留40岁，重算1330元", "after_answer")], [q()],
        ["更正后517元失效；不得重复追问年龄或被保人"], [quote(40,True,after="initial_answer"),quote(40,False)])
    add("MT", "高血压含糊保字", "我有高血压，可以保吗？", {"goal":"能否投保","disease":"高血压"},
        [reply(["goal"],"我问的是能不能买，不是已经买了能不能赔。",{"goal":"能否投保"},"区分购买与理赔，不能用既往症免责代替核保结论", "if_requested")], [handoff()],
        ["允许澄清保字，也允许直接说明承保需人工核实；不能为凑轮数强制索取血压/用药等医学字段", "有依据的提前安全转人工视为有效处理，此例不算完成多轮能力证明"])
    add("MT", "缺年龄后确认父亲超龄", "我给父亲第一次买这个必选计划，年龄合适吗？", {"age":72},
        [reply(["age"],"父亲72周岁。",{"age":72},"首次投保超70岁，不把续期费率当新客资格")],
        [obligation("eligibility","72岁不在首次投保年龄范围；不能因费率表有72岁档就说能首次购买",["age","first_age"])], ["只补判断当前问项所需年龄，不必收集无关报价条件"])

    add("CP", "35岁资格价格优惠全部条件齐", "我35岁有职工医保，给自己第一次买必选计划，年缴个人单标准价多少钱？年龄能投保吗？我作为同一投保人，和配偶、女儿共3人首次买适用系列，同一订单且没有旧人数，有什么家庭优惠？", {}, [],
        [eligibility(),q(),discount(3)], ["必须分别回应三项；已有条件不重复追问；标准保费与家庭折扣口径分开说明"], [quote(35,True)])
    add("CP", "年缴月缴及全年差额", "我30岁有社保，自己首次买必选计划个人单，年缴和月缴各多少钱？月缴12个月比年缴多多少？", {}, [],
        [q("报318元/年、31元/月，全年372元，差额54元"),obligation("difference","31*12-318=54元，不把31元乘12说成318元",["first_age"])],
        ["不能只报年缴或月缴，必须交代比较结果"], [quote(30,True),quote(30,True,"month"),difference(30,True)])
    add("CP", "一般住院和ICU津贴", "我首次买基础必选计划，一般住院免赔是多少？进ICU每天给多少津贴？要另外买什么吗？", {}, [],
        [obligation("deductible","一般医疗年免赔1万元",["deductible"]),obligation("icu","ICU津贴500元/天，需选购相应住院津贴加油包，不是基础必选自带",["icu_allowance"])],
        ["同时覆盖免赔、ICU金额、加油包归属，不能只提其中两项"])
    add("CP", "普通门诊与住院免赔关系", "基础版普通门急诊和一般住院各是多少免赔？两个各扣一万，还是共享？", {}, [],
        [obligation("shared","一般门急诊与一般医疗共享年免赔1万元，不是各扣一万；按各自责任条件适用",["shared_basic"])], ["普通门诊不得混成门急诊加油包规则"])
    add("CP", "家庭优惠和共享免赔不是一回事", "我同一投保人和配偶、女儿3人首次买适用系列，同一订单无旧人数。家庭优惠多少？免赔自动共享吗？普通门急诊也算家庭共享吗？", {}, [],
        [discount(3),obligation("family_deductible","家庭共享免赔需相应加油包；同一投保人同一订单一般医疗共享1万元，普通门急诊不在该家庭累计范围",["family_deductible"])],
        ["家庭优惠不等于自动买了共享免赔加油包；不混淆基础版两责任共享与家庭跨人共享"])
    add("CP", "药省保多个限制", "给孩子买药省保，等待多久？报多少比例，每月每年限额是多少？孩子6周岁能开处方药吗？", {}, [],
        [obligation("drug_waiting","等待期7天",["drug_waiting"]),obligation("drug_limit","保险人认可的互联网药店合规药品费用50%，月100元、年1000元",["drug_ratio","drug_limit"]),obligation("prescription","6周岁含以下无法开处方药；不能说满6岁就可开",["drug_prescription"])],
        ["四个问项及年龄含边界都检查，不能因提到药就泛化为全部药能赔"])
    add("CP", "56岁不同计划年龄边界", "我56岁第一次买，基础必选计划年龄能买到吗？门急诊加油包能一起首次买吗？", {}, [],
        [obligation("basic_age","56岁在基础计划首次投保年龄范围，其他条件仍须满足",["age"]),obligation("addon_age","门急诊加油包首次年龄上限55岁，56岁不能首次买；不得拿续期档位解释新客资格",["age","first_age"])],
        ["两种计划分开判断，不能一概拒绝或一概接受"])
    add("CP", "72岁首次投保资格和费用", "父亲72岁有社保，第一次买必选计划，能投保吗？年缴多少钱？", {}, [],
        [obligation("eligibility","明确首次投保超过70岁上限",["age"]),obligation("price","说明不能按首次投保报价；若解释费率表71岁以上档，必须指出指定期限重新投保口径，不能作为父亲可购报价",["first_age"])],
        ["不能直接用表中3928元作为72岁首次报价，价格问项不能被静默忽略"])
    add("CP", "病史购买与理赔分开", "我已经有高血压，现在第一次买能通过吗？以后因为高血压住院能报吗？", {}, [],
        [handoff(),obligation("claim_scope","说明既往症相关费用的免责规定及保单特别约定，不能承诺买了即可赔高血压住院",["preexisting"])],
        ["核保和理赔两个问项分别交代，不能把不能确认核保变成任何费用都不赔"])
    add("CP", "已知报价与缺附件混合", "我30岁有社保，首次买必选计划，个人单年缴多少钱？原位癌一定属于合同100种重疾吗？某个特效药一定在清单里吗？", {}, [],
        [q(),obligation("disease_definition","没有逐项重疾定义附件，不能确认原位癌，需人工核实",["missing_attachments"]),obligation("drug_list","没有完整药品附件，不能确认某药，需人工核实并保留这项问题",["missing_attachments"])],
        ["有依据的318元报价应答；两个待核实事项均进入人工请求，不能只存一个"], [quote(30,True)])

    add("MX", "用户原例35岁三问", "我现在35岁能投保吗？投保需要多少钱，有优惠吗？", {"has_social":True,"plan":"必选计划","payment":"year","family_count":3,"qualified_family":True},
        [reply(["relation","has_social","plan","payment"],"给自己第一次买必选计划，按年交，有职工医保，先报个人单标准价。",{"relation":"本人","first_insurance":True,"plan":"必选计划","payment":"year","has_social":True},"记得还有资格与优惠两问"),
         reply(["family_count","qualified_family"],"同一投保人名下自己、配偶、孩子共3人首次买适用系列，同一订单，没有旧人数。",{"family_count":3,"qualified_family":True},"资格条件、423元标准年费、10%家庭优惠都覆盖")],
        [eligibility(),q(),discount(3)], ["条件齐前可以给完整条件分支，不能擅自按有医保单人报价", "回答补充信息后必须回到原始三个问项"], [quote(35,True)])
    add("MX", "缺年龄医保的报价与等待期", "自己第一次买必选计划，年缴和月缴各多少钱？生病住院和意外住院都要等30天吗？", {"age":40,"has_social":False},
        [reply(["age"],"40周岁。",{"age":40},"已知等待期先答，不把整题转人工"),reply(["has_social"],"没有社保。",{"has_social":False},"年1330/月131，同时完成等待期区分")],
        [q(),wait()], ["价格条件缺失不影响已有等待期依据；后续不能丢月缴或意外问项"], [quote(40,False),quote(40,False,"month")])
    add("MX", "高血压核保和价格缺医保", "我35岁有高血压，第一次给自己买必选计划，个人单年缴多少钱？病史能让我投保吗？", {"has_social":False},
        [reply(["has_social"],"没有医保。",{"has_social":False},"报价971元，核保转人工并携带病史")],
        [q(),handoff()], ["不能因核保待核实直接终止并丢失报价；若直接转交完整原问及缺医保事项，记安全转交但不算自动完成报价", "明确价格只是标准费率，不等于高血压已被承保"], [quote(35,False)])
    add("MX", "母亲年缴月缴与差额", "我给母亲第一次买必选计划，个人单一年多少钱、每月多少钱？一年总共差多少？", {"age":62,"has_social":True},
        [reply(["age"],"母亲62岁。",{"age":62},"保留三个价格问项"),reply(["has_social"],"她有居民医保。",{"has_social":True},"2563元年缴，253元月缴，全年3036元，多473元")],
        [q(),obligation("difference","253*12-2563=473元",["first_age"])], ["不能用投保人的年龄；补医保后不只回答月缴"], [quote(62,True),quote(62,True,"month"),difference(62,True)])
    add("MX", "测算缺条件加等待期", f"一般住院合规费用15000元，没走医保，{scope}。能测算赔多少？另问首次投保生病和意外都等30天吗？", {"has_social":True,"remaining_deductible":10000},
        [reply(["has_social"],"按有医保身份投保。",{"has_social":True},"保留没走医保，适用60%"),reply(["remaining_deductible"],"免赔一万元还完全没用。",{"remaining_deductible":10000},"测算3000元并区分疾病意外等待期")],
        [obligation("calculation","假设下(15000-10000)*60%=3000元，不能按100%测算",["deductible","no_social_settlement"]),wait()],
        ["测算与等待期两项都完成，允许先答等待期再补计算条件"], [claim(15000,10000,"0.6")])
    add("MX", "家庭人数不明与共享范围", "家庭买有多少优惠？一般住院免赔能共享吗，门急诊也能一起算吗？", {"family_count":4,"qualified_family":True},
        [reply(["family_count","qualified_family"],"同一投保人名下自己、配偶和两个孩子共4人，首次买适用系列，同一订单，没有旧人数。",{"family_count":4,"qualified_family":True},"优惠15%，共享需加油包，普通门急诊不累计")],
        [discount(4),obligation("family_deductible","共享须相应加油包，同一订单一般医疗年免赔1万元；普通门急诊不计入家庭累计",["family_deductible"])], ["不能把家庭优惠人数当免赔额，也不能要求医保后才回答共享规则"])
    add("MX", "复合报价后更正医保", "我35岁有医保，首次买必选计划个人单，年缴和月缴各多少钱？全年差多少？", {"corrected_has_social":False},
        [reply([],"我刚说错了，没有医保，刚才三个价格都按无医保重算一下。",{"has_social":False},"年971/月96/差181元，旧423/42/81元失效", "after_answer")],
        [q(),obligation("difference","更正前42*12-423=81元；更正后96*12-971=181元",["first_age"])],
        ["前后两轮分别核对三个问项，不能只改年缴"], [quote(35,True,after="initial_answer"),quote(35,True,"month",after="initial_answer"),difference(35,True,after="initial_answer"),quote(35,False),quote(35,False,"month"),difference(35,False)])
    add("MX", "已有家庭成员累计和本人价格", "我首次买必选计划，自己30岁，个人单年缴多少钱？同一个投保人名下已有配偶一张适用系列保单，我这次新增自己，家庭优惠是多少？", {"has_social":True},
        [reply(["has_social"],"有职工医保，这次只新增我，没有其他家人。",{"has_social":True,"family_count":2,"qualified_family":True},"318元标准年费，累计2人5%，不能只看本次1人")],
        [q(),discount(2)], ["现有配偶与本次本人累计；不重复问已经明确的年龄/家庭关系"], [quote(30,True)])
    add("MX", "超龄父亲换母亲后补医保", "父亲72岁第一次买，能买吗，年缴多少？", {"new_relation":"母亲","age":65,"has_social":True},
        [reply([],"那改成母亲65岁第一次买必选计划，给她报个人单年缴，父亲不买。",{"relation":"母亲","age":65},"换人后不能继承父亲年龄和医保", "after_answer"),
         reply(["has_social"],"母亲有居民医保。",{"has_social":True},"母亲年龄符合范围，标准年缴2563元")],
        [obligation("father_eligibility","先说明72岁父亲超首次上限，不给可购报价",["age","first_age"]),obligation("mother_eligibility","更换为65岁母亲后重新按其条件判断，年龄在范围，仍以核保等为准",["age","underwriting"]),q()],
        ["因父亲超龄不应直接挂断阻止用户换人；不能把父亲问项算母亲已答", "这一例初始没有披露任何人的医保"], [quote(65,True)])
    add("MX", "长口语背景和五人优惠", "我之前刷到广告，家里一直商量，平时工作忙没来得及看。我35岁，想给自己第一次买必选计划，一年要多少钱？年龄能不能买？家里好几个人还想一起买，优惠怎么算？", {"has_social":True,"family_count":5,"qualified_family":True},
        [reply(["has_social"],"我有职工医保，先报个人单年缴标准价。",{"has_social":True,"payment":"year"},"不丢年龄资格和优惠问项"),
         reply(["family_count","qualified_family"],"我是同一个投保人，这次自己、配偶、孩子和父母共5人，首次买适用系列，同一订单，没有旧人数。",{"family_count":5,"qualified_family":True},"给423元年费、年龄条件、累计5人20%优惠")],
        [eligibility(),q(),discount(5)], ["长背景不能掩盖三个问题；不因用户说商量就触发挽留而跳过问答"], [quote(35,True)])


def dataset() -> dict:
    return {"schema_version": "1.0", "created_at": "2026-10-13", "status": "not_run",
            "purpose": "复杂问题定义对应的完整对话材料；非官方测试集，非实测成绩",
            "execution": {
                "session": "每个用例新建同一持续会话，空画像、空历史；可标准开场，不注入隐藏信息",
                "agent_input_allowlist": ["initial_user", "执行中实际披露的user_reply_rules.utterance"],
                "operator_only": ["hidden_user_facts", "user_reply_rules", "obligations", "conversation_checks", "numeric_oracles", "source_registry"],
                "reply_policy": "操作者或隔离用户模拟器按坐席真实追问提供相关回复；合并问多个字段时可合并对应回复。无关字段继续隐藏。after_answer更正在正式答复后主动发送，不能因第一轮已答而漏掉。if_requested分支允许安全提前转交。",
                "all_replies_meaning": "仅指该数字所需条件均已披露，不要求发送无关或可跳过的回复",
                "no_question": "缺信息又不追问时记录失败，不擅自喂齐条件；允许条件分支先答，但需取得用户条件后落地或明确待处理",
                "early_end": "必答事项未解决且未完整转交就结束，记失败，不另开会话强行续测",
                "medical_limits": "不要求无核保规则下收集特定医学指标，不设置血压阈值或自行推断可保；允许有依据的安全转人工",
                "diagnostic_round_cap": 12, "round_cap_note": "仅执行防死循环上限，非赛题指标；到上限仍未完成记未解决",
                "runner": "人工执行或隔离用户模拟器按本协议适配CallSession；旧exp21单轮runner不能直接读取本文件",
            },
            "grading": {
                "per_obligation_states": ["answered_correctly", "clarification_pending", "safely_handed_off", "wrong", "omitted"],
                "pass": "所有问项正确回答或已完整安全转交，逐轮检查点通过，实际结论有原文依据且无编造；追问本身不算原问题完成",
                "numeric_policy": "金额必须绑定被保人、医保、计划、年/月、投保口径和披露阶段，不以数字出现即通过；金额须与oracle一致，可接受等价中文表述",
                "handoff": "按既有用户认可口径另记安全处理；请求必须保存问项和已知条件，未报价部分若转交须明列，不能声称人工已接通",
                "reports": ["按三类及dev/evaluation_candidate分别统计", "多轮最终解决率与逐轮追问/记忆/更正正确性", "复合问项完整率与整例正确率", "自动解决、转人工、未解决分别报告", "数字/来源/动作错误分别记录"],
                "timing": "文字用例只记逐轮文字就绪耗时；语音适配另录用户句尾、首声、正式首声、实播结束。沿用现有≤3秒/≤10秒口径，不用整通电话时长替代单轮时延，不把追问当完整回答",
                "voice_limit": "材料本身未验证ASR长句拆分、真人打断、回声或电话远端；需另走语音验收",
            },
            "split_policy": "每类前6例dev，后4例evaluation_candidate，共18/12；均在2026-10-13编写。涉及已知失败模式，不能称盲测。评估后若用于调优，标记为dev并另建未使用留出集。",
            "material_baseline": "source_registry为当前本地资料快照；资料补齐核保规则、健康告知或完整附件后，重审对应转人工预期再升级版本",
            "source_registry": SOURCES, "cases": CASES}


def validate(data: dict) -> None:
    ids = [c["id"] for c in data["cases"]]
    assert len(ids) == len(set(ids)) == 30
    assert Counter(c["type"] for c in data["cases"]) == {"multi_turn":10,"compound":10,"mixed":10}
    assert all(c["split"] in {"dev", "evaluation_candidate"} for c in data["cases"])
    for sid, s in data["source_registry"].items():
        raw = (ROOT / s["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == s["sha256"], f"Changed source: {sid}"
        assert s["excerpt"] in raw.decode("utf-8-sig").splitlines()[s["line"]-1], sid
    for c in data["cases"]:
        assert c["initial_profile"] == {} and c["initial_history"] == []
        assert c["status"] == "not_run" and c["initial_user"] and c["obligations"] and c["conversation_checks"]
        assert len({o["id"] for o in c["obligations"]}) == len(c["obligations"])
        assert c["type"] == "compound" or c["user_reply_rules"]
        for r in c["user_reply_rules"]:
            assert r["mode"] in {"when_requested","after_answer","if_requested"}
            assert r["utterance"] and r["reveals"] and r["checkpoint"]
        for o in c["obligations"] + c["numeric_oracles"]:
            assert o["source_ids"] and all(s in data["source_registry"] for s in o["source_ids"])
        for oracle in c["numeric_oracles"]:
            values = oracle["inputs"]
            if oracle["kind"] == "standard_premium":
                fresh = quote(values["age"], values["has_social"], values["payment"], oracle["after"])
                assert fresh == oracle, c["id"]
                assert values["age"] <= 70, "First-insurance oracle exceeds age limit"
            elif oracle["kind"] == "annual_payment_difference":
                amount = Decimal(str(values["monthly"])) * values["months"] - Decimal(str(values["annual"]))
                assert amount == Decimal(str(oracle["expected"])), c["id"]
            elif oracle["kind"] == "illustrative_claim":
                amount = max(Decimal(values["eligible_net_expense"]-values["remaining_deductible"]), Decimal(0)) * Decimal(values["ratio"])
                assert amount == Decimal(str(oracle["expected"])), c["id"]
            else:
                raise ValueError("Unknown oracle")


def render(data: dict) -> str:
    out = ["# 复杂问题测试集与用例", "", "编写日期：2026-10-13。本文件与JSON保留编写时的用例定义，JSON中的not_run为冻结状态，实际运行记录另存。", "",
           "**后续执行状态**：exp42已执行30例首轮真实文字测试，共49次交互；仍有漏问、漏答及错误假设，3例操作者触发需人工复测。结果见[完整历史接入与验证](26-完整历史默认接入与超限压缩.md)，不重算既有准确率、时延或达标状态。", "",
           "对应[复杂问题定义](02-命题解决思路.md#111-复杂问题的定义)。机器可读文件：[complex_dialogue_v1.json](../experiments/data/complex_dialogue_v1.json)。", "",
           "## 范围与分组", "", "共30例：多轮交互MT、单轮多问题CP、两类混合MX各10例。每类前6例开发，后4例评估候选，共18/12。候选涉及已知失败模式，不称盲测；一旦用于调优需调整标记并另留新题。", "",
           "## 执行与判分", "", "每例使用独立的持续会话，从空画像和空历史开始，只发送初始用户原话。隐藏条件、回复规则、标准要点和数字答案由操作者或隔离用户模拟器持有，不能提前交给坐席。", "",
           "按真实追问逐项披露条件；若一次问多个字段，可以合并对应用户回复。不得在坐席没有追问时主动喂齐条件来掩盖失败。标记“正式回答后主动更正”的回复必须执行；标记“仅在追问时”的回复允许有依据的安全转人工提前结束该分支。每个回复后检查条件保留和未答问项，不能只评最后一句。", "",
           "每个问项分别标记：正确回答、追问待处理、安全转交、错误、遗漏。整例通过须所有问项有正确回答或完整安全转交，逐轮检查点通过，结论有依据且无编造。安全转交要保存完整问题、已确认条件及未完成问项；未答完也未转交就结束记失败。自动解决、安全转交和未解决分别统计，不能把追问当原问题完成。", "",
           "数字答案只在其必要条件披露后适用；前一阶段答案与更正后答案分别核对。出现正确数字不足以通过，还须对象、医保、计划、年/月和首次投保口径正确。条件分支说明可以先给，补条件后须落到用户实际情形。核保规则缺失时允许直接说明并安全转交，不强制索取无依据的医学指标。", "",
           "文字执行记录每轮原话、全文回答、实际模型输入/画像、事件、出处及人工请求；语音执行增加音频和用户句尾、首声、正式首声、实播结束时间。沿用当前时延口径，整通咨询完成情况与单轮≤3秒/≤10秒分别报告。防死循环上限12轮仅为执行限制，未完成记未解决。", "",
           "本文件不能直接喂给旧exp21单轮脚本。需要人工逐例执行，或按JSON的回复触发规则适配隔离用户模拟器与当前CallSession。材料不证明真人打断、ASR长句切分或电话远端能力。", "",
           "## 用例总览", "", "| 编号 | 类型 | 分组 | 场景 |", "|---|---|---|---|"]
    labels = {"multi_turn":"多轮交互","compound":"单轮多问题","mixed":"两类混合"}
    for c in data["cases"]:
        out.append(f"| {c['id']} | {labels[c['type']]} | {'开发' if c['split']=='dev' else '评估候选'} | {c['title']} |")
    for c in data["cases"]:
        out += ["", f"## {c['id']}：{c['title']}", "", f"**初始用户**：{c['initial_user']}", ""]
        if c["hidden_user_facts"]:
            out += ["**操作者隐藏条件**：`"+json.dumps(c["hidden_user_facts"],ensure_ascii=False)+"`。", ""]
        if c["user_reply_rules"]:
            out += ["**后续用户回复与检查点**：", ""]
            for i, r in enumerate(c["user_reply_rules"],1):
                when = {"when_requested":"追问对应条件时","after_answer":"正式回答后主动更正","if_requested":"仅在追问对应事项时"}[r["mode"]]
                fields = "、".join(r["requested_fields"])
                out.append(f"{i}. {when}{'（'+fields+'）' if fields else ''}：“{r['utterance']}” 检查：{r['checkpoint']}。")
            out.append("")
        out += ["**最终必答问项**：", ""]
        for o in c["obligations"]:
            out.append(f"- {o['id']}：{o['expected']}。依据：{', '.join(o['source_ids'])}。")
        if c["numeric_oracles"]:
            out += ["", "**数值核验**：", ""]
            for o in c["numeric_oracles"]:
                out.append(f"- {o['kind']} / {o['after']}：{json.dumps(o['inputs'],ensure_ascii=False)} → **{o['expected']:g}{o['unit']}**。")
        out += ["", "**对话检查**：", ""]
        out += [f"- {check}。" for check in c["conversation_checks"]]
    out += ["", "## 资料依据与复核", "", "金额直接读本地原始费率整理表；规则附当前原文位置及SHA-256快照。健康告知、核保规则和完整附件缺口以当前资料目录为准；若后续补齐须重审转人工预期。来源指向实际原文，不只给模型的标准答案。", ""]
    for sid, s in data["source_registry"].items():
        out.append(f"- `{sid}`：[原文 {s['path']}:{s['line']}](../{s['path']})；摘录：{s['excerpt']}。")
    out += ["", "离线校验（不调用云模型、不改结果）：", "", "```powershell", "python experiments/build_complex_dialogue_v1.py --check", "```", "",
            "校验30例唯一编号、三类分布、分组、来源文件与行号、资料哈希、数字费率与测算一致性；不代表坐席通过业务测试。编辑JSON后需同步用例文档，勿直接重新生成覆盖人工修订。", ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read saved material; no file writes or cloud calls")
    args = parser.parse_args()
    if args.check:
        data = json.loads(DATA.read_text(encoding="utf-8"))
        validate(data)
        assert DOC.read_text(encoding="utf-8") == render(data), "Document and JSON are out of sync"
    else:
        if DATA.exists() or DOC.exists():
            raise SystemExit("Material already exists; use --check. Do not overwrite reviewed cases.")
        setup_sources()
        build_cases()
        data = dataset()
        validate(data)
        DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        DOC.write_text(render(data), encoding="utf-8")
    print(json.dumps({"cases":len(data["cases"]), "by_type":dict(Counter(c["type"] for c in data["cases"])),
                      "by_split":dict(Counter(c["split"] for c in data["cases"])),
                      "material_check":"passed", "business_test":"not_run"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
