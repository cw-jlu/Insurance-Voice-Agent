"""Coverage facts as fixed sentences (方案 A) and as checkable values (方案 B).

方案 A "只选不写": the answer model gets numbered fact sentences and cites them as [F3]; the program puts the
sentence in. Each sentence names its own coverage item, so a rule cannot drift to another item.
方案 B "确定性比对": a small model lists the (item, field, value) claims an answer makes; the program compares
them with the coverage table. The table, not a model, decides whether a claim is right.
Both are built from the coverage table (保障计划表) and its clause cells; see 08 根因分析与改进方案.
"""

from __future__ import annotations

import json
import re

from agent import cloud
from agent.knowledge import coverage_items

COVERAGE = {i["id"]: i for i in coverage_items()}

SHORT = {
    "R01": "一般医疗（住院）", "R02": "一般门急诊", "R03": "重大疾病医疗", "R04": "特定疾病康复住院",
    "R05": "恶性肿瘤先进疗法（质子重离子等）", "R06": "特定药品（癌症和罕见病特药）", "R07": "重疾异地转诊交通住宿",
    "R08": "重疾住院护工", "R09": "家庭共享免赔额", "R10": "重疾关爱一次性给付", "R11": "重疾特需医疗",
    "R12": "住院医疗费用补偿", "R13": "门急诊加油包", "R14": "一般住院津贴", "R15": "ICU住院津贴", "R16": "药省保（网上买药）",
}


def _pack(iid: str) -> str:
    return COVERAGE[iid]["责任&加油包"].replace("尊享", "")


def is_addon(iid: str) -> bool:
    return COVERAGE[iid]["责任&加油包"] != "必选责任"


def item_facts(iid: str) -> list[str]:
    """Fact sentences for one coverage item, each naming the item. Written from the table and clause cells."""
    n = SHORT[iid]
    out = [f"{n}属于{_pack(iid)}，需要在基础版之外另外选购" if is_addon(iid) else f"{n}是基础版就有的保障"]
    special = {
        "R01": ["一般医疗和一般门急诊共享300万的年度额度", "一般医疗和一般门急诊每年共享1万元免赔额",
                "一般医疗扣除免赔额后按100%赔；按有社保身份投保、但看病没用社保结算的，按60%赔",
                "一般医疗包括住院、特殊门诊、门诊手术和住院前后30天的门急诊费用"],
        "R02": ["一般门急诊和一般医疗共享300万的年度额度", "一般门急诊和一般医疗每年共享1万元免赔额",
                "一般门急诊扣除免赔额后按100%赔；按有社保身份投保、但看病没用社保结算的，按60%赔"],
        "R03": ["重大疾病医疗一年最高300万", "重大疾病医疗没有免赔额", "重大疾病医疗按100%赔",
                "重大疾病医疗限合同约定的100种重大疾病"],
        "R04": ["特定疾病康复住院一年最高100万", "特定疾病康复住院没有免赔额",
                "特定疾病康复住院在指定康复医疗机构按80%赔，在二级及以上公立医院或指定民营医院按60%赔",
                "特定疾病康复住院的康复转运救护车单次最多1000元"],
        "R05": ["恶性肿瘤先进疗法一年最高600万", "恶性肿瘤先进疗法没有免赔额", "恶性肿瘤先进疗法按100%赔",
                "恶性肿瘤先进疗法不包括化疗、靶向等药品费"],
        "R06": ["特定药品一年最高600万", "特定药品没有免赔额",
                "特定药品要在指定药店买，用药前要先申请处方审核",
                "特定药品若是社保目录内的药，社保报销后剩余部分按100%赔；目录外的药按100%赔"],
        "R07": ["重疾异地转诊交通住宿一年最多1万元", "重疾异地转诊交通住宿没有免赔额", "重疾异地转诊的住宿每天最多500元"],
        "R08": ["重疾住院护工每天500元", "重疾住院护工一次最多30天、一年最多30天", "重疾住院护工一年最多1.5万元"],
        "R09": ["家庭共享免赔额让同一订单的家庭成员共用1万元免赔额，只限一般医疗，一般门急诊不参与共享"],
        "R10": ["重疾关爱一次性给付是初次确诊100种重疾时给5万元"],
        "R11": ["重疾特需医疗一年最高300万", "重疾特需医疗没有免赔额", "重疾特需医疗按100%赔",
                "重疾特需医疗可在特需部、国际部、VIP部，床位费每天最多1500元"],
        "R12": ["住院医疗费用补偿一年最高1万元", "住院医疗费用补偿没有免赔额",
                "住院医疗费用补偿按100%赔，但痔疮、结节、息肉、囊肿、增生和女性生殖系统疾病按20%赔"],
        "R13": ["门急诊加油包一年最多1万元", "门急诊加油包按50%赔，每次最多300元",
                "门急诊加油包按有社保身份投保、但没用社保结算的按30%赔", "门急诊加油包只能55周岁及以下投保"],
        "R14": ["一般住院津贴每天100元，每次住院前3天不给", "一般住院津贴一次住院最多给30天、一年最多60天"],
        "R15": ["ICU住院津贴每天500元，从第一天就给", "ICU住院津贴一次最多给30天、一年最多60天",
                "ICU住院津贴可以和一般住院津贴同时拿"],
        "R16": ["药省保按50%赔，每月最多100元、一年最多1000元", "药省保限指定的互联网药店"],
    }[iid]
    out += special
    wait = COVERAGE[iid]["等待期"]
    if iid in ("R01", "R02", "R03", "R04"):
        out.append(f"{n}的等待期是{wait}，意外伤害导致的不受等待期限制")
    else:
        out.append(f"{n}的等待期是{wait}")
    return out


# ---------------------------------------------------------------- 方案 B: claims vs table
TABLE = {  # field -> item -> allowed values (numbers; % as plain numbers; amounts in 元; days)
    "免赔额": {"R01": {10000}, "R02": {10000}, "R03": {0}, "R04": {0}, "R05": {0}, "R06": {0}, "R07": {0},
               "R11": {0}, "R12": {0}},
    "保额": {"R01": {3e6}, "R02": {3e6}, "R03": {3e6}, "R04": {1e6}, "R05": {6e6}, "R06": {6e6}, "R07": {1e4},
             "R08": {1.5e4}, "R10": {5e4}, "R11": {3e6}, "R12": {1e4}, "R13": {1e4}, "R14": {6000}, "R15": {3e4}, "R16": {1000}},
    "赔付比例": {"R01": {100, 60}, "R02": {100, 60}, "R03": {100, 60}, "R04": {80, 60, 48, 36}, "R05": {100, 60},
                 "R06": {100, 60}, "R07": {100}, "R11": {100}, "R12": {100, 20}, "R13": {50, 30}, "R16": {50}},
    "等待期": {i: ({7} if i == "R16" else {30}) for i in COVERAGE},
    "日额": {"R08": {500}, "R14": {100}, "R15": {500}, "R07": {500}, "R11": {1500}},
}

_CLAIM_SYS = """从保险 AI 的回答中列出它对各项责任说的具体规则，只输出 JSON：{"claims": [{"item": 责任编号, "field": 字段, "value": 原话}]}
字段只能是：免赔额、保额（最多报多少）、赔付比例、等待期、日额（每天给多少）、归属（基础版就有 / 要另外买加油包）。
回答里把哪项责任说成什么，就按那项责任记，不要纠正；"住院"指一般医疗 R01，"门诊/急诊"指 R02，"大病/重疾/癌症医疗费"指 R03。
示例：
回答"一般医疗和重疾医疗都0免赔" → [{"item":"R01","field":"免赔额","value":"0"},{"item":"R03","field":"免赔额","value":"0"}]
回答"家庭共享免赔额基础版就有" → [{"item":"R09","field":"归属","value":"基础版就有"}]
回答"住院一年最多报600万" → [{"item":"R01","field":"保额","value":"600万"}]
回答"ICU津贴要另外买住院津贴加油包，每天500元" → [{"item":"R15","field":"归属","value":"要另外买"},{"item":"R15","field":"日额","value":"500元"}]
没有具体规则就输出 {"claims": []}。
责任目录：
{catalog}"""


def _catalog() -> str:
    return "\n".join(f"{i} {SHORT[i]}（{COVERAGE[i]['责任名称']}，{COVERAGE[i]['责任&加油包']}）" for i in COVERAGE)


def _value(v) -> float | None:
    s = str(v).strip()
    if re.search(r"没有|无|不设|不用|0元|^0$|零", s) and not re.search(r"\d*[1-9]", s):
        return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*(万)?", s)
    if not m:
        return None
    x = float(m.group(1)) * (10000 if m.group(2) else 1)
    return x


_TYPE_OK = {
    "保额": lambda x: bool(re.search(r"\d.*(万|元)|^[一二三四五六七八九十百千]+万", x)) and not re.search(r"天|次|月|%", x),
    "免赔额": lambda x: (bool(re.search(r"\d.*(万|元)", x)) or bool(re.fullmatch(r"\s*(0|零|没有|无|不设|0元)\s*", x)))
              and not re.search(r"天|次|月|%", x),
    "赔付比例": lambda x: "%" in x and not re.search(r"天|元", x),
    "等待期": lambda x: bool(re.fullmatch(r"\s*\d+\s*天\s*", x)),
    "日额": lambda x: "元" in x and not re.search(r"万|%", x),
}


def check_claims(answer: str, client=None) -> list[str]:
    """Problems found by comparing the answer's claims with the coverage table (empty list: nothing wrong found)."""
    raw = cloud.chat([{"role": "system", "content": _CLAIM_SYS.replace("{catalog}", _catalog())}, {"role": "user", "content": answer}],
                     model=cloud.FAST_MODEL, client=client, temperature=0, response_format={"type": "json_object"})
    try:
        claims = json.loads(raw).get("claims", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    problems = []
    for c in claims if isinstance(claims, list) else []:
        iid, field = c.get("item"), c.get("field")
        if iid not in COVERAGE:
            continue
        if field == "归属":
            said_addon = bool(re.search(r"加油包|另外|加购|单独|额外", str(c.get("value"))))
            said_base = bool(re.search(r"基础版|自带|不用另外|包含", str(c.get("value"))))
            # only the harmful direction: an add-on described as included in the base plan
            if said_base and not said_addon and is_addon(iid):
                problems.append(f"{SHORT[iid]}是加油包，需要另外选购，不是基础版就有")
            continue
        raw = str(c.get("value"))
        # type check (第十轮消融: 8 of 9 blocks were mis-extractions such as "保额 = 30天", "等待期 = 确诊后180天内")
        if not _TYPE_OK.get(field, lambda x: True)(raw):
            continue
        allowed = TABLE.get(field, {}).get(iid)
        v = _value(raw)
        if allowed is None or v is None:
            continue
        if not any(abs(v - a) < 1e-6 for a in allowed):
            right = "、".join(f"{a / 10000:g}万" if a >= 10000 else f"{a:g}" for a in sorted(allowed))
            problems.append(f"{SHORT[iid]}的{field}说成了{c.get('value')}，应为{right}")
    return problems
