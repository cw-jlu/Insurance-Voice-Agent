"""Small-model classifiers that replace the rule-only versions (see 06 实验记录 第三轮).

Each function has a rule fast path for unambiguous inputs and calls the fast model otherwise,
so the common cases stay instant and the long tail is still handled.

The model behind the classification tasks is switchable: tasks listed in ``JEV_TASKS`` (env, comma-separated:
barge, identity, gaps, gap_review, act) go to Jev, which returns probabilities instead of text; the rest use
``cloud.FAST_MODEL``. Slot extraction and fact check generate structured text and always use the LLM.
"""

from __future__ import annotations

import json
import os
import re

from agent import cloud
from agent.dialogue import _BACKCHANNEL, _STOP
from agent.guardrail import cn_to_int

JEV_TASKS = {t.strip() for t in os.getenv("JEV_TASKS", "").split(",") if t.strip()}
# probability of "yes" at or above which a Jev noul answer counts as yes
JEV_THRESHOLDS = {"barge": 0.5, "identity": 0.5, "gaps": 0.5, "gap_review": 0.5}

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
    if "barge" in JEV_TASKS:
        return ("打断" if jev_barge_in(t) >= JEV_THRESHOLDS["barge"] else "附和"), "jev"
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
    if "identity" in JEV_TASKS:
        return jev_identity(text) >= JEV_THRESHOLDS["identity"]
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
    if "gap_review" in JEV_TASKS:
        return jev_gap_review(question, answer) >= JEV_THRESHOLDS["gap_review"]
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
    if "act" in JEV_TASKS:
        return jev_act(text, history)["choice"]
    raw = cloud.chat([{"role": "system", "content": _ACT_SYS},
                      {"role": "user", "content": (f"最近对话：\n{history}\n" if history else "") + f"用户：{text}"}],
                     model=cloud.FAST_MODEL, client=client, temperature=0, response_format={"type": "json_object"})
    try:
        act = json.loads(raw).get("act", "提问")
    except json.JSONDecodeError:
        act = "提问"
    return act if act in ACTS else "提问"


# --- data-gap detection on the CURRENT utterance only ---------------------------------------------

_GAPQ_SYS = """判断用户这一句话（只看这一句，不考虑之前的对话）是否在问以下资料里没有的事项，只输出 JSON：{"gaps": [...]}
可选：
- "疾病核保"：用户说自己或家人有某种病、做过手术、在吃药，问能不能买、能不能承保、会不会影响投保、买了能不能报这个病；
- "续保"：问能不能续保、保证续保、停售后能否续、老了能不能续、明年还能不能买；
- "社保抵扣免赔额"：问医保/社保报销的部分能不能抵扣或计入免赔额，医保报完剩下的怎么报、还能不能报；
- "条款清单"：问重疾清单、特药清单、指定医院名单等具体清单。
都不是就输出 {"gaps": []}。问保障范围、价格、等待期、免赔额是多少，都不算。"""


def detect_gaps(text: str, client=None) -> list[str]:
    if "gaps" in JEV_TASKS:
        return [g for g, p in jev_gaps(text).items() if p >= JEV_THRESHOLDS["gaps"]]
    raw = cloud.chat([{"role": "system", "content": _GAPQ_SYS}, {"role": "user", "content": text}],
                     model=cloud.FAST_MODEL, client=client, temperature=0, response_format={"type": "json_object"})
    try:
        g = json.loads(raw).get("gaps", [])
    except json.JSONDecodeError:
        return []
    return [x for x in g if x in ("疾病核保", "续保", "社保抵扣免赔额", "条款清单")]


# --- Jev versions (typed questions; each returns probabilities so thresholds can be tuned) ---------

_J_BARGE = ("AI 坐席正在说话时，用户插了这句话。用户是想让 AI 停下来吗？"
            "叫停（等一下）、纠正自己的信息（我没有社保）、提出新问题（那月缴多少）、拒绝（我不需要）都算想让 AI 停下；"
            "只是表示在听、同意或让对方继续（嗯、对、好的、然后呢、你说、原来如此、明白了）不算。")


def jev_barge_in(text: str) -> float:
    """Probability that the utterance is a real interruption (not a backchannel)."""
    return cloud.jev(f"用户：{text}", {"q": {"type": "noul", "instructions": _J_BARGE}})["q"]["noul"]


_J_ID = ("用户是在问和他通话的是不是真人、AI、机器人或录音吗？"
         "例如\"你是机器人吗\"\"你是语音助手吧\"\"对面是人吗\"\"这是录音吗\"算；"
         "要求转人工或找真人来说、索要通话录音、问有没有某种 AI 服务、问机器人手术，都不算。")


def jev_identity(text: str) -> float:
    return cloud.jev(f"用户：{text}", {"q": {"type": "noul", "instructions": _J_ID}})["q"]["noul"]


_J_GAPREV = ("这是保险电销 AI 对用户问题的回答。资料里没有以下内容，AI 不允许下结论："
             "某种疾病、病史、手术或用药的人能不能投保、能不能承保、会不会影响投保；能不能续保、是否保证续保；"
             "医保/社保报销的部分能不能抵扣或计入免赔额。"
             "AI 的回答是否对上述问题给出了肯定或否定的结论（例如\"可以买\"\"不影响\"\"一般能通过核保\"\"保证续保\"\"可以抵扣\"）？"
             "说\"以核保/条款为准\"\"要做健康告知\"\"帮您转专员\"、说明既往症及相关费用不赔、提醒如实告知，都不算下结论。")


def jev_gap_review(question: str, answer: str) -> float:
    """Probability that the answer gives a conclusion on a data gap."""
    return cloud.jev(f"用户问题：{question}\nAI 回答：{answer}", {"q": {"type": "noul", "instructions": _J_GAPREV}})["q"]["noul"]


_J_GAPS = {
    "疾病核保": "用户是否说自己或家人有某种病、做过手术或在吃药，并问能不能买、能不能承保、会不会影响投保、买了能不能报这个病？",
    "续保": "用户是否在问能不能续保、是否保证续保、停售后能否续、老了能不能续、明年还能不能买？",
    "社保抵扣免赔额": "用户是否在问医保/社保报销的部分能不能抵扣或计入免赔额，或医保报完剩下的怎么报、还能不能报？",
    "条款清单": "用户是否在问重疾清单、特药清单、指定医院名单等具体清单？",
}
_J_GAPS_NOTE = "只看用户这一句话。问保障范围、价格、等待期、免赔额是多少，都不算。"


def jev_gaps(text: str) -> dict[str, float]:
    """Probability per gap topic; the four questions are evaluated in one request."""
    qs = {f"g{i}": {"type": "noul", "instructions": f"{q}{_J_GAPS_NOTE}"} for i, q in enumerate(_J_GAPS.values())}
    ans = cloud.jev(f"用户：{text}", qs)
    return {g: ans[f"g{i}"]["noul"] for i, g in enumerate(_J_GAPS)}


_J_ACTS = {
    "提问": "问保障、价格、规则等；一句话里既有信息又有提问时也选这个",
    "提供信息": "回答坐席的问题，或补充、纠正自己的情况（年龄、社保、给谁买、病史）",
    "附和": "嗯、好的、你说",
    "同意投保": "愿意买、要链接、问怎么买、让发过来",
    "拒绝": "不需要、没兴趣、别打了、不买",
    "异议": "嫌贵、要考虑、已经有医保或别的保险、不信任",
    "要人工": "要求转人工或真人",
    "投诉": "表达不满、要投诉",
    "身份": "问是不是真人、机器人、AI、录音",
}


def jev_act(text: str, history: str = "") -> dict:
    """Returns {"choice": act, "confidence": float, "probabilities": {act: p}}."""
    state = (f"最近对话：\n{history}\n" if history else "") + f"用户（当前这句）：{text}"
    a = cloud.jev(state, {"act": {"type": "choice", "instructions": "用户当前这句话在保险电销对话中的作用是什么？",
                                  "criteria": _J_ACTS}})["act"]
    if a.get("choice") not in ACTS:
        a["choice"] = "提问"
    return a
