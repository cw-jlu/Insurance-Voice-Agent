"""Small-model classifiers that replace the rule-only versions (see 06 实验记录 第三轮).

Each function has a rule fast path for unambiguous inputs and calls the fast model otherwise,
so the common cases stay instant and the long tail is still handled.
"""

from __future__ import annotations

import json
import re

from agent import cloud
from agent.dialogue import _BACKCHANNEL, _STOP
from agent.guardrail import cn_to_int

# --- barge-in --------------------------------------------------------------------------------------

BARGE_LABELS = ["附和", "打断"]
_BARGE_SYS = """你在判断电话中用户的一句插话。AI 坐席正在说话时，用户说了一句话。
判断这句话是"附和"还是"打断"：
- 附和：只是表示在听、同意、让对方继续，例如 嗯、对、好的、然后呢、你说、原来如此、明白了。AI 应该继续说。
- 打断：用户想让 AI 停下，包括 叫停（等一下）、纠正信息（我没有社保）、提出新问题（那月缴多少）、拒绝（我不需要）。
只输出两个字：附和 或 打断。"""


def classify_barge_in(text: str, client=None) -> tuple[str, str]:
    """Returns (label, path) where path is "rule" or "model"."""
    t = text.strip()
    if not t or _BACKCHANNEL.match(t):
        return "附和", "rule"
    if _STOP.search(t):
        return "打断", "rule"
    out = cloud.chat([{"role": "system", "content": _BARGE_SYS}, {"role": "user", "content": t}],
                     model=cloud.FAST_MODEL, client=client, temperature=0, max_tokens=4).strip()
    return ("附和" if "附和" in out else "打断"), "model"


# --- slot extraction -------------------------------------------------------------------------------

_SLOT_SYS = """从用户一句话中抽取投保报价需要的条件，只输出 JSON：
{"age": 被保人年龄整数或null, "has_social": true/false/null, "pay": "年缴"/"月缴"/null,
 "relation": "本人"/"配偶"/"子女"/"父母"/null, "gender": "男"/"女"/null}
规则：只填用户明确说出的；"六十九了""快5了""今年14""四十一"都是年龄；
"单位给交社保""有城乡医保"表示有社保，"社保停了""断了""没买过"表示没有社保；
"我妈"→relation=父母、gender=女；"我爱人"性别不明时 gender=null。"""


def extract_slots(text: str, client=None) -> dict:
    raw = cloud.chat([{"role": "system", "content": _SLOT_SYS}, {"role": "user", "content": text}],
                     model=cloud.FAST_MODEL, client=client, temperature=0, response_format={"type": "json_object"})
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    out = {}
    if isinstance(d.get("age"), (int, float)):
        out["age"] = int(d["age"])
    elif isinstance(d.get("age"), str):
        v = int(d["age"]) if d["age"].isdigit() else cn_to_int(d["age"])
        if v is not None:
            out["age"] = v
    for k in ("has_social", "pay", "relation", "gender"):
        if d.get(k) is not None:
            out[k] = d[k]
    return out


# --- identity --------------------------------------------------------------------------------------

_ID_SYS = """判断用户这句话是不是在问"和他通话的是不是真人/AI/机器人/录音"。
是：例如"你是机器人吗""你是语音助手吧""对面是人吗""这是录音吗"。
不是：要求转人工或找真人来说、索要通话录音、问有没有某种AI服务、问机器人手术。只输出：是 或 否。"""


def is_identity_question(text: str, client=None) -> bool:
    out = cloud.chat([{"role": "system", "content": _ID_SYS}, {"role": "user", "content": text}],
                     model=cloud.FAST_MODEL, client=client, temperature=0, max_tokens=2).strip()
    return out.startswith("是")


# --- gap / conclusion judge (model review of an answer) -------------------------------------------

_GAP_SYS = """你在审核保险电销 AI 的回答。以下内容资料里没有，AI 不允许给出结论：
1. 某种疾病（或病史、手术、用药）的人能不能投保、能不能承保、会不会影响投保；
2. 能不能续保、是否保证续保、停售后能否续保；
3. 医保/社保报销的部分能不能抵扣或计入免赔额。
如果回答对上述问题给出了肯定或否定的结论（例如"可以买""不影响""一般能通过核保""保证续保""可以抵扣"），输出 违规；
以下属于合规（资料里有依据）：说"以核保/条款为准""要做健康告知""帮您转专员"；说明"既往症（投保前已有的病）及其相关费用不赔"；提醒如实告知；或问题本身与上述无关。
只输出两个字：违规 或 合规。"""


def gap_review(question: str, answer: str, client=None) -> bool:
    """True if the answer gives a conclusion it is not allowed to give."""
    out = cloud.chat([{"role": "system", "content": _GAP_SYS},
                      {"role": "user", "content": f"用户问题：{question}\nAI 回答：{answer}"}],
                     model=cloud.FAST_MODEL, client=client, temperature=0, max_tokens=4).strip()
    return "违规" in out


# --- fact check: does each sentence attribute rules to the right coverage item? ----------------------

_FACT_SYS = """你在核对保险 AI 的回答是否和资料一致。资料按【组】给出，每组属于一项责任或一份文件。
逐句检查回答：
1. 句中提到的金额、比例、天数、免赔额、共享范围、限制条件，是否能在资料中找到，并且属于句子所说的那项责任（不能把甲责任的规则说成乙责任的）；
2. 是否有资料里没有的说法。
只输出 JSON：{"ok": true/false, "problems": ["有问题的句子及原因", ...]}。没有问题时 problems 为空列表。
措辞不同、省略部分信息不算问题，只有说错或说了资料里没有的内容才算。"""


def fact_check(answer: str, grouped_evidence: str, client=None) -> list[str]:
    raw = cloud.chat([{"role": "system", "content": _FACT_SYS},
                      {"role": "user", "content": f"【资料】\n{grouped_evidence}\n\n【回答】{answer}"}],
                     model=cloud.FAST_MODEL, client=client, temperature=0, response_format={"type": "json_object"})
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [] if d.get("ok", True) else [str(p) for p in d.get("problems", [])][:3]


# --- dialogue act (for the sales flow) ------------------------------------------------------------

ACTS = ["提问", "提供信息", "附和", "同意投保", "拒绝", "异议", "要人工", "投诉", "身份"]
_ACT_SYS = """判断电话里用户这句话在销售对话中的作用，只输出 JSON：{"act": 以下之一}
提问：问保障、价格、规则等；提供信息：回答坐席的问题或补充、纠正自己的情况（年龄、社保、给谁买、病史）；
附和：嗯、好的、你说；同意投保：愿意买、要链接、怎么买、发给我；拒绝：不需要、没兴趣、别打了、不买；
异议：嫌贵、要考虑、已经有医保或别的保险、不信任；要人工：要求转人工或真人；投诉：表达不满、要投诉；
身份：问是不是真人、机器人、AI、录音。
一句话里既有信息又有提问时选"提问"。"""


def classify_act(text: str, history: str = "", client=None) -> str:
    raw = cloud.chat([{"role": "system", "content": _ACT_SYS},
                      {"role": "user", "content": (f"最近对话：\n{history}\n" if history else "") + f"用户：{text}"}],
                     model=cloud.FAST_MODEL, client=client, temperature=0, response_format={"type": "json_object"})
    try:
        act = json.loads(raw).get("act", "提问")
    except json.JSONDecodeError:
        act = "提问"
    return act if act in ACTS else "提问"
