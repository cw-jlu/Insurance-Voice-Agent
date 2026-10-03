"""Slow-lane answering pipeline: understand → route → assemble evidence → generate → guard.

Routing follows the solution design (02 方案 3.4):
- prices come from the quote engine, never from the LLM;
- coverage facts come from the coverage table (R01–R16);
- high-frequency key topics map to fixed clauses (no retrieval);
- everything else goes through semantic + keyword retrieval;
- gap questions get a fixed fallback for the part the data cannot answer.

Answers the program can give alone are not sent to the LLM at all (第六轮 80 题错题分析: the LLM's own
additions after a correct quote or gap reply were wrong in 5 of 13 errors): quotes (including price change
and add-on prices), the waiting period, and renewal / clause-list gap questions use templates.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from agent import cloud
from agent.guardrail import (
    GAPS, IDENTITY_REPLY, compliance_guard, extract_numbers, gap_guard, gap_of, is_identity_question, number_guard,
    social_deduction_conclusion,
)
from agent.knowledge import Chunk, clause_chunks, coverage_items
from agent.knowledge import rate_table
from agent.quote import ADDONS, MAX_AGE_NEW, Insured, quote
from agent.retrieval import Retriever

COVERAGE = {i["id"]: i for i in coverage_items()}

# --- fixed citations for key topics (curated from 投保须知 / 责任免除 / 费率表 text) ------------------
TOPICS: dict[str, list[tuple[str, str]]] = {
    "总保额": [("投保须知", "国内医疗总保额为600万元")],
    "既往症": [("责任免除", "既往症或先天性疾病，不予理赔"), ("责任免除", "因既往症，性病")],
    "等待期": [("投保须知", "必选计划的等待期为30天"), ("责任免除", "等待期内被保险人确诊疾病"),
             ("保障计划表", "被保险人因遭受意外伤害事故或在等待期30天后因罹患疾病")],
    "补偿原则": [("责任免除", "补偿原则")],
    "医院范围": [("责任免除", "非中国公立二级"), ("投保须知", "二级及二级以上的公立医院及本保单约定的指定民营医疗机构普通部发生的")],
    "社保结算比例": [("责任免除", "未以基本医疗保险或公费医疗身份就诊并结算的，则保险人根据本合同单独约定的给付比例")],
    "被保人年龄": [("投保须知", "被保险人年龄")],
    "投保人资格": [("投保须知", "投保人资格")],
    "职业": [("投保须知", "被保险人职业类别")],
    "一人一份": [("投保须知", "限投保1份")],
    "生效时间": [("投保须知", "次日零时")],
    "保险期间": [("投保须知", "本产品保险期间")],
    "价格变化": [("投保须知", "随着您的年龄增长而变化")],
    "如实告知": [("投保须知", "如实告知义务")],
    "其他免责": [("责任免除", "整形、美容、怀孕"), ("责任免除", "精神和行为障碍"), ("责任免除", "职业病或医疗事故")],
}
TOPIC_NAMES = list(TOPICS) + ["家庭单优惠"]

FAMILY_RULE = ("家庭单优惠：同一投保人名下尊享医疗险系列和众民保医疗险系列保单的家庭成员（本人、配偶、子女、父母、配偶父母）"
               "人数合计2人每人优惠5%，3人优惠10%，4人优惠15%，5人及以上优惠20%；与优选体优惠同时满足时按家庭单优惠。")

GAP_NAMES = [g.name for g in GAPS] + ["条款清单"]
GAP_FALLBACK = {g.name: g.fallback for g in GAPS}
# The old text promised "我可以把条款发给您", which the system cannot do (第六轮 2 calls).
GAP_FALLBACK["条款清单"] = "具体清单以保险条款为准，我这边可以帮您转专员确认。"
GAP_FALLBACK["社保抵扣免赔额"] = "医保已经报销的部分不会重复赔；医保报销的钱能不能算进1万免赔额，要以条款为准，我帮您转专员确认。"
# gaps answered with a template only; 疾病核保 and 社保抵扣 questions usually carry an answerable part too
TEMPLATE_ONLY_GAPS = {"续保", "条款清单"}

INTENTS = ["报价", "优惠", "保障", "投保条件", "免责", "缺口", "身份", "闲聊", "其他"]


def _find(doc: str, needle: str) -> list[Chunk]:
    return [c for c in clause_chunks() if c.doc == doc and needle in c.text]


def topic_chunks(topic: str) -> list[Chunk]:
    out: list[Chunk] = []
    for doc, needle in TOPICS.get(topic, []):
        for c in _find(doc, needle):
            if c not in out:
                out.append(c)
    return out


def _ded_label(it: dict) -> str:
    """The table's "免赔额/免赔天数" column holds the daily payout for 护工/津贴 items, not a deductible."""
    v = it["免赔额/免赔天数"]
    return "给付标准" if ("元/天" in v and "护工" in it["责任名称"]) or "日津贴" in v else "免赔额"


def item_chunks(item_id: str) -> list[Chunk]:
    it = COVERAGE[item_id]
    name = it["责任名称"]
    head = Chunk(
        chunk_id=f"{item_id}:head",
        doc="保障计划表",
        text=(f"{name}（{it['责任&加油包']}）：人群{it['人群']}；等待期{it['等待期']}；保额{it['保额']}；"
              f"赔付比例{'；'.join(it['赔付比例'])}；{_ded_label(it)}{it['免赔额/免赔天数']}；医院范围{'；'.join(it['医院范围'])}"),
        source=f"保障计划表 第{it['source_rows'][0]}–{it['source_rows'][1]}行",
        meta={"item": item_id},
    )
    cells = [c for c in clause_chunks() if c.meta.get("item") == item_id]
    return [head] + cells


# --- hybrid retriever with embeddings ------------------------------------------------------------

class HybridRetriever:
    """TF-IDF (char n-gram) + embedding cosine, fused by reciprocal rank."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, client=None):
        self.chunks = chunks
        self.lex = Retriever(chunks)
        self.emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.client = client

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Chunk]:
        n = len(self.chunks)
        rankings = []
        if mode in ("hybrid", "lexical"):
            lex = self.lex.search(query, k=n, method="tfidf")
            pos = {c.chunk_id: r for r, c in enumerate(lex)}
            rankings.append([pos[c.chunk_id] for c in self.chunks])
        if mode in ("hybrid", "dense"):
            q = np.array(cloud.embed([query], client=self.client)[0])
            q /= np.linalg.norm(q)
            sims = self.emb @ q
            order = np.argsort(-sims)
            rank = np.empty(n, dtype=int)
            rank[order] = np.arange(n)
            rankings.append(list(rank))
        fused = np.zeros(n)
        for r in rankings:
            fused += 1.0 / (60 + np.array(r) + 1)
        return [self.chunks[i] for i in np.argsort(-fused)[:k]]


# --- understanding -------------------------------------------------------------------------------

_UNDERSTAND_SYS = """你是保险电销外呼对话的理解模块，只输出 JSON，不要解释。
产品：尊享e生·百万医疗保险2026版。根据用户最新一句话（结合已知条件），输出：
{
 "intent": 以下之一 %(intents)s,
 "slots": {"age": 整数或null, "has_social": true/false/null, "pay": "年缴"/"月缴"/null,
           "gender": "男"/"女"/null, "relation": "本人"/"配偶"/"子女"/"父母"/null, "family_size": 整数或null},
 "items": 相关保险责任编号列表，从下面目录中选，最多 %(max_items)s 个；一句话问到几项就列几项，只列直接相关的,
 "topics": 相关关键主题列表，从 %(topics)s 中选,
 "gaps": 用户问的是否属于数据无法回答的内容，从 %(gaps)s 中选，没有则为空列表,
 "rewrite": 把用户口语改写成保险条款用语的检索查询（例如"以前就有的病"→"既往症"）
}
规则：
- 问价格、多少钱、每月多少 → intent=报价。
- 用户说自己（或家人）患有某种病、做过手术、在吃药，问能不能买/能不能报 → intent=缺口，gaps 含"疾病核保"，topics 含"既往症"和"如实告知"。
- 只是问某类疾病或费用本身在不在保障范围（如职业病、先天性疾病、精神疾病、美容、牙科、分娩、体检），没有说自己患病 → intent=免责，topics 含"其他免责"或"既往症"，gaps 为空。
- 问总共保多少、保额多少、能报什么 → topics 含"总保额"。
- 问优惠、打折、便宜点、全家几口人一起买 → intent=优惠，topics 含"家庭单优惠"（不是报价，不需要年龄）。
- 比较按月交和按年交、问明年/以后价格 → intent=报价。"明年X岁"表示现在 X-1 岁，age 填现在的年龄。
- 用户只是补充或修改条件（如"有社保呢""我35岁""按月交呢""不对，我没有社保"）→ 沿用【上一句】的意图。
- "得了某病能报吗""某病的费用报不报"（没说自己已经患病）→ intent=免责；只有"我有/我妈有某病，能不能买、买了能不能报"才是疾病核保缺口。
- 问到期怎么续、明年还能不能买、老了能不能续 → gaps 含"续保"。
- 问医保报完/报过的部分还能不能报、医保报完剩下的怎么报、和医保怎么配合 → topics 含"补偿原则"；只有问医保/社保报销的钱能不能算进或抵扣1万免赔额时，gaps 才含"社保抵扣免赔额"。
- 问没走社保、没用医保卡结算能报多少（比例）→ topics 含"社保结算比例"，gaps 为空（资料有规定）。
- 问某种病（如癌症、白血病、心脏搭桥）能不能报、怎么报，没说自己或家人已经有这个病 → gaps 为空。
- 问等待期、刚买就生病、意外要不要等 → topics 含"等待期"。
- 问免赔额以内、没到1万、自己先掏的部分能不能报 → items 含 R01、R12（住院医疗费用补偿加油包）、R09（家庭共享免赔额加油包）。
- 问确诊重病能不能直接拿钱、一次性给钱 → items 含 R10（重疾关爱加油包）和 R03。
- 问职业（如高空作业、司机、矿工）能不能买 → intent=投保条件，topics 含"职业"，gaps 为空。
- 问能不能多买几份、重复投保 → intent=投保条件，topics 含"一人一份"。
- 问续保、停售 → gaps 含"续保"；问医保报销部分能否抵扣/算进免赔额 → gaps 含"社保抵扣免赔额"；问疾病清单、药品清单、医院名单 → gaps 含"条款清单"。
- 问是不是真人/机器人/AI → intent=身份。
- 年龄：用户说"我妈65岁"时 age=65、relation=父母、gender=女。只填用户明确说出的条件，不要猜。
责任目录：
%(catalog)s"""


def _catalog() -> str:
    return "\n".join(f"{i['id']} {i['责任名称']}（{i['责任&加油包']}）" for i in coverage_items())


MAX_ITEMS = 5  # exp14: listing up to 5 items in one pass replaces the evidence agent (same accuracy, P50 -2.2 s)


def understand(question: str, known: dict, client=None, model: str | None = None, max_items: int | None = None) -> dict:
    max_items = max_items or MAX_ITEMS
    sys = _UNDERSTAND_SYS % {"intents": INTENTS, "topics": TOPIC_NAMES, "gaps": GAP_NAMES, "catalog": _catalog(),
                             "max_items": max_items}
    prev = known.get("_prev_utterance")
    history = known.get("_history")
    shown = {k: v for k, v in known.items() if not k.startswith("_")}
    user = (f"已知条件：{json.dumps(shown, ensure_ascii=False)}\n"
            + (f"【最近对话】\n{history}\n" if history else "")
            + (f"【上一句】用户：{prev}（上一句意图：{known.get('_prev_intent')}）\n" if prev and not history else "")
            + f"用户：{question}")
    raw = cloud.chat([{"role": "system", "content": sys}, {"role": "user", "content": user}],
                     model=model or cloud.FAST_MODEL, client=client, temperature=0,
                     response_format={"type": "json_object"})
    try:
        u = json.loads(raw)
    except json.JSONDecodeError:
        u = {"intent": "其他", "slots": {}, "items": [], "topics": [], "gaps": [], "rewrite": question}
    u.setdefault("slots", {})
    u["items"] = [i for i in u.get("items", []) if i in COVERAGE][:max_items]
    u["topics"] = [t for t in u.get("topics", []) if t in TOPIC_NAMES]
    u["gaps"] = [g for g in u.get("gaps", []) if g in GAP_NAMES]
    return u


# --- generation ----------------------------------------------------------------------------------

_ANSWER_SYS = """你是众安保险的 AI 电销坐席，正在电话里给用户介绍尊享e生·百万医疗保险2026版。
要求：
1. 只根据【资料】回答。资料里没有的内容不要说，不要推测，不要用常识补充。
2. 回答里的每个数字（金额、比例、天数、年龄）都必须在【资料】或【报价结果】里原样出现。价格只能用【报价结果】里的数字。
3. 口语化，先说结论，一般不超过 100 字，最多讲 3 个要点；不要 Markdown、不要列表符号、不要编号。
4. 讲保障时顺带说一个最重要的限制（免赔额、等待期、医院范围或既往症）。
5. 禁止说"什么病都能报""保证续保""肯定能赔"之类的承诺；涉及病史要提醒如实告知，能否投保以核保结果为准，不要替核保下结论。
6. 如果【必须原样包含的话】不为空，把它自然地放进回答里。
7. 如果相关责任属于加油包（资料里写着"××加油包"），要说明这是需要另外选购的加油包，不在基础版里。
8. 资料按【组】给出，每组只属于一项责任或一份文件。每句话只能依据同一组资料，不同责任的规则（免赔额、共享范围、比例、天数）不能互相套用。
9. 说赔付比例时，如果资料里还有"未以社保身份就诊结算"的比例，要一并说明。注意：这个降低后的比例只适用于"按有社保身份投保、但看病没用社保结算"的情况；按无社保投保的人，扣除免赔额后仍按资料中的正常比例赔付。
10. 资料里没有理赔流程、理赔材料、理赔时效、APP 或线上操作方式，被问到就说以条款为准、可以转专员，不要描述流程。
11. 不要说"抱歉""我重新说""刚才说错了"之类的话，直接回答。
12. 如果基础版不保、但资料里某个加油包保（例如确诊重疾一次性给付、免赔额以内的住院费用），要分开说：基础版怎样，另外选购××加油包后可以怎样。不要说基础版没有资料里明明有的责任。"""


@dataclass
class Answer:
    text: str
    route: str
    understanding: dict = field(default_factory=dict)
    evidence: list[Chunk] = field(default_factory=list)
    structured: set = field(default_factory=set)
    violations: list[str] = field(default_factory=list)
    attempts: int = 0
    fallback_used: bool = False
    timings: dict = field(default_factory=dict)


def grouped_context(evidence: list[Chunk]) -> str:
    """Evidence grouped by coverage item (or by document), each group labelled with what it belongs to."""
    groups: dict[str, list[Chunk]] = {}
    for c in evidence:
        key = c.meta.get("item") or c.doc
        groups.setdefault(key, []).append(c)
    out = []
    for key, cs in groups.items():
        if key in COVERAGE:
            it = COVERAGE[key]
            label = f"{it['责任名称']}（{'基础版必选责任' if it['责任&加油包'] == '必选责任' else it['责任&加油包'] + '，需另外选购'}）"
        else:
            label = key
        out.append(f"【组：{label}】\n" + "\n".join(f"- {c.text}（出处：{c.source}）" for c in cs))
    return "\n".join(out)


def _guard(question: str, text: str, evidence: list[Chunk], structured: set, gaps: list[str] | None = None,
           fact_check: bool | str = "multi", gaps_judged: bool = False) -> list[str]:
    v = number_guard(text, [c.text for c in evidence], structured).violations
    # When the caller already judged gaps (CallSession: current-sentence detector), only those gaps are
    # enforced; the keyword rule alone misfires on e.g. "得了癌症……能报吗" (第六轮重测).
    v += (gap_guard("", text, gaps) if gaps_judged else gap_guard(question, text, gaps)).violations
    # "医保报完剩下的扣1万再报" is a conclusion on the 社保抵扣 gap whatever the question was (第六轮 2 calls)
    v += social_deduction_conclusion(text).violations
    v += compliance_guard(text).violations
    from agent import small_model

    items = {c.meta.get("item") for c in evidence if c.meta.get("item")}
    run_fact = fact_check is True or fact_check == "always" or (fact_check == "multi" and len(items) >= 2)
    jobs = {}
    with ThreadPoolExecutor(2) as ex:  # the two model checks are independent: run them side by side
        if gaps:
            jobs["gap"] = ex.submit(small_model.gap_review, question, text)
        if run_fact and evidence and not v:
            jobs["fact"] = ex.submit(small_model.fact_check, text, grouped_context(evidence))
    if "gap" in jobs and jobs["gap"].result():
        v.append("模型复核：对缺口问题给出了结论")
    if "fact" in jobs:
        v += [f"事实核对：{p}" for p in jobs["fact"].result()]
    return v


def _quote_text(slots: dict) -> tuple[str, set] | None:
    age, social = slots.get("age"), slots.get("has_social")
    if age is None or social is None:
        return None
    y = quote(Insured(int(age), bool(social), gender=slots.get("gender")), "年缴")
    m = quote(Insured(int(age), bool(social), gender=slots.get("gender")), "月缴")
    if not y.ok:
        return (f"{int(age)}岁超过了首次投保的年龄上限{MAX_AGE_NEW}周岁，这款产品买不了。" if int(age) > MAX_AGE_NEW
                else "出生满30天才能投保。"), {float(age), float(MAX_AGE_NEW), 30.0}
    who = {"父母": "您家老人", "子女": "孩子", "配偶": "您爱人"}.get(slots.get("relation") or "", "您")
    txt = (f"{who}{int(age)}岁{'有' if social else '没有'}社保，基础版一年{y.total:g}元，"
           f"按月交每月{m.total:g}元。")
    return txt, y.numbers() | m.numbers() | {round(m.total * 12, 2), float(age)}


def _item_template(item_ids: list[str]) -> str:
    """Fallback built straight from coverage-table fields, so every number comes from the table."""
    parts = []
    for i in item_ids[:2]:
        it = COVERAGE[i]
        addon = "" if it["责任&加油包"] == "必选责任" else f"这是{it['责任&加油包']}里的责任，需要另外选购；"
        parts.append(f"{addon}{it['责任名称']}保额{it['保额']}，赔付比例{'、'.join(it['赔付比例'])}，"
                     f"免赔额{it['免赔额/免赔天数']}，等待期{it['等待期']}。")
    return "".join(parts)


# --- program-generated answers ---------------------------------------------------------------------

_PRICE_Q = re.compile(r"多少钱|几块|价格|价钱|保费|费用多少|一年(要|交)?多少|每月|每个月|一个月|月缴|年缴|按月|按年|交多少|"
                      r"贵|便宜|优惠|打折|报价|怎么收费")
_ELIGIBLE_Q = re.compile(r"(还)?(能|可以|能不能)(买|投|投保)|多大(年纪|岁数)?(能|可以)?(买|投)|有没有(什么)?(年龄)?限制|年纪.{0,4}限制")


def _eligibility_text(slots: dict, question: str) -> str:
    age = int(slots["age"])
    if age > MAX_AGE_NEW:
        return f"{age}岁超过了首次投保的年龄上限{MAX_AGE_NEW}周岁，这款产品买不了。"
    tail = "门急诊加油包只能55周岁及以下买。" if age > 55 else ""
    return f"{age}岁可以买，首次投保的年龄是出生满30天到{MAX_AGE_NEW}周岁。{tail}能不能承保还要看健康告知和核保结果。"


_DISCOUNT_Q = re.compile(r"优惠|打折|折扣|便宜|全家|家里人|一家|一起买")
FAMILY_SAY = "优惠的话，给家人一起买有家庭单优惠：2个人每人优惠5%，3个人10%，4个人15%，5个人及以上20%。"
_PRICE_CHANGE_Q = re.compile(r"以后|每年|明年|后年|将来|涨|一直(都)?(是)?这个价|固定|变不变|会变|老了")
_WAIT_Q = re.compile(r"等待期|等多久|等多长|多久(才)?(能|可以)?(用|生效|报|赔)|刚买就|买完就|马上就?(生病|住院)|意外.{0,10}(等|等待)|要等")
WAIT_TEXT = ("疾病的等待期是30天，从保单生效算起，等待期内确诊的疾病不赔；意外受伤导致的住院和门急诊不受这30天限制。"
             "加油包里药省保加油包的等待期是7天。")
_ADDON_ALIASES = {
    "家庭共享免赔额加油包": r"家庭共享|共享免赔",
    "重疾关爱加油包": r"重疾关爱|重疾险|确诊.{0,4}(拿|给)钱",
    "重大疾病扩展特需医疗加油包": r"特需|VIP|vip|国际部",
    "住院医疗费用补偿加油包": r"费用补偿|补偿加油包|住院补偿",
    "门急诊医疗加油包": r"门急诊加油包|门诊加油包|门急诊医疗加油包",
    "住院津贴加油包": r"津贴|ICU|icu",
    "药省保加油包": r"药省保|网上买药|互联网药",
}
_COVERAGE_WORDS = re.compile(r"能报|报不报|赔|保障|免赔|等待|医院|比例|怎么报|保什么|能保")


def addons_in(text: str) -> list[str]:
    return [a for a, pat in _ADDON_ALIASES.items() if re.search(pat, text)]


def _who(slots: dict) -> str:
    return {"父母": "您家老人", "子女": "孩子", "配偶": "您爱人"}.get(slots.get("relation") or "", "您")


def _price_change_text(slots: dict) -> tuple[str, set] | None:
    """Prices go by age band: say the current band and the next one (both from the rate table)."""
    age, social = slots.get("age"), slots.get("has_social")
    if age is None or social is None or int(age) > MAX_AGE_NEW:
        return None
    tbl, col = rate_table("年缴_必选"), "必选计划_有社保" if social else "必选计划_无社保"
    p, band = tbl.price(int(age), col)
    nxt = next((b for b in tbl.bands if band and b[0] > band[1]), None)
    if p is None or nxt is None or nxt[0] > MAX_AGE_NEW:
        return None
    np_, _ = tbl.price(nxt[0], col)
    txt = (f"不是固定的，保费每年按当时的年龄段算。{_who(slots)}现在{int(age)}岁在{band[0]}到{band[1]}岁这一档，基础版一年{p:g}元；"
           f"到{nxt[0]}岁进入下一档，一年{np_:g}元。")
    return txt, {float(int(age)), float(band[0]), float(band[1]), float(nxt[0]), float(nxt[1]), p, np_}


def _addon_quote_text(slots: dict, addons: list[str]) -> tuple[str, set] | None:
    age, social = slots.get("age"), slots.get("has_social")
    if age is None or social is None:
        return None
    parts, nums = [], {float(int(age))}
    for a in addons:
        r = quote(Insured(int(age), bool(social), gender=slots.get("gender"), addons=[a]), "年缴")
        line = next((l for l in r.lines if a in l.label), None)
        if line is None:
            if r.needs:
                parts.append(f"{a}的价格男女不一样，请问被保人是男士还是女士？")
            else:
                limit = re.search(r"上限(\d+)周岁", "".join(r.errors))
                parts.append(f"{a}只能{limit.group(1)}周岁及以下投保，{int(age)}岁不能选。" if limit else f"{a}在这个年龄不能选。")
                if limit:
                    nums.add(float(limit.group(1)))
            continue
        unit = "每5万保额" if a == "重疾关爱加油包" else ""
        parts.append(f"{a}需要在基础版之外另外选购，{_who(slots)}{int(age)}岁{unit}一年{line.amount:g}元。")
        nums.add(line.amount)
    return "".join(parts), nums


QUESTION_MARKERS = re.compile(r"吗|呢|多少|多久|几|什么|怎么|哪|能不能|是不是|可不可以|有没有|咋|？|\?")


def _clauses(text: str) -> int:
    return len([c for c in re.split(r"[，,。；;？?]", text) if QUESTION_MARKERS.search(c)])


def addon_notice(text: str, evidence: list[Chunk]) -> str:
    """If the answer talks about an add-on item but never says it must be bought separately, say so."""
    out = text
    for c in evidence:
        iid = c.meta.get("item")
        if not iid or COVERAGE[iid]["责任&加油包"] == "必选责任":
            continue
        it = COVERAGE[iid]
        key = it["责任名称"].replace("保险金", "")[:6]
        pack = it["责任&加油包"]
        if (key in out or pack in out or pack.replace("尊享", "") in out) and not re.search(r"另外|额外|单独|加购|附加|自选", out):
            out = out.rstrip() + f"{pack.replace('尊享', '')}需要在基础版之外另外选购。"
            break
    return out


def combine_gaps(understood: list[str], detected: list[str]) -> list[str]:
    """Gaps the answer is routed on. A gap counts only when the understanding module and the gap detector
    agree (exp12: each alone misfires on different questions, e.g. "确诊癌症能直接拿到钱吗"); the detector
    names the type. Two exceptions, where one path is enough: clause lists on the understanding module (it does
    not misfire on them), and 社保抵扣免赔额, whose fallback is true either way (it states the 补偿原则)."""
    out = list(detected) if understood else []
    for g, either in (("条款清单", "条款清单" in understood), ("社保抵扣免赔额", "社保抵扣免赔额" in understood + detected)):
        if either and g not in out:
            out.append(g)
    return out


def gaps_to_check(understood: list[str], detected: list[str]) -> list[str]:
    """Gaps the finished answer is checked against: either path is enough (checking costs little, missing costs a lot)."""
    return list(dict.fromkeys(understood + detected))


class Pipeline:
    def __init__(self, retriever: HybridRetriever | None, client=None, slow_model: str | None = None,
                 fast_model: str | None = None, fact_check: str = "off"):
        self.fact_check = fact_check  # "always" | "multi" (only when evidence spans 2+ coverage items) | "off"
        self.retriever = retriever
        self.client = client or cloud.llm_client()
        self.slow_model = slow_model or cloud.SLOW_MODEL
        self.fast_model = fast_model or cloud.FAST_MODEL

    def retrieve(self, question: str, k: int = 4) -> list[Chunk]:
        return self.retriever.search(question, k=k) if self.retriever is not None else []

    def answer(self, question: str, known: dict | None = None, voice: bool = False,
               agent_steps: int = 0, agent_budget_s: float = 3.0, understanding: dict | None = None) -> Answer:
        """``known`` carries the call's profile; ``known["_history"]`` (recent turns, as text) and
        ``known["_prev_utterance"]`` come from CallSession. ``agent_steps`` > 0 lets the evidence
        agent add tool calls on top of the fixed routing, within ``agent_budget_s`` seconds."""
        known = dict(known or {})
        t0 = time.perf_counter()
        timings: dict = {}

        if is_identity_question(question):
            return Answer(f"{IDENTITY_REPLY}价格和保障我都能帮您讲清楚，您想先了解哪方面？", route="身份(规则)",
                          timings={"total": time.perf_counter() - t0})

        if understanding is None:
            from agent import small_model

            with ThreadPoolExecutor(3) as ex:
                fu = ex.submit(understand, question, known, self.client, self.fast_model)
                fd = ex.submit(small_model.detect_gaps, question, self.client)
                fr = ex.submit(self.retrieve, question)
                u, detected = fu.result(), fd.result()
                u["_retrieved"] = fr.result()
            u["_gaps_check"] = gaps_to_check(u.get("gaps", []), detected)
            u["gaps"], u["_gaps_judged"] = combine_gaps(u.get("gaps", []), detected), True
            if u.get("intent") == "缺口" and not u["gaps"]:
                u["intent"] = "保障" if u.get("items") else "其他"
        else:
            u = understanding
        timings["understand"] = time.perf_counter() - t0
        new_slots = {k: v for k, v in (u.get("slots") or {}).items() if v is not None}
        slots = {**known, **new_slots}
        intent = u.get("intent", "其他")
        # "不对，我没有社保" / "那我爸61岁有医保呢": the user supplied or corrected a price condition and asked nothing
        # else, so the answer is the (new) price (80 题 N076, N079; the call flow does the same in _advance).
        if (intent not in ("报价", "缺口", "身份") and ({"age", "has_social"} & set(new_slots))
                and slots.get("age") is not None and slots.get("has_social") is not None
                and not _COVERAGE_WORDS.search(question)):
            intent = u["intent"] = "报价"

        if intent == "身份":
            from agent import small_model

            if small_model.is_identity_question(question, self.client):
                return Answer(f"{IDENTITY_REPLY}价格和保障我都能帮您讲清楚，您想先了解哪方面？", route="身份(模型)",
                              understanding=u, timings={**timings, "total": time.perf_counter() - t0})
            intent = u["intent"] = "其他"  # only one model thought so (第八轮验收 4 calls, 1 question)

        evidence: list[Chunk] = []
        structured: set = {float(v) for v in slots.values() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        structured |= {n.value for n in extract_numbers(question)}  # numbers the user said ("住院10天") may be repeated
        must_say: list[str] = []
        route = intent

        if intent == "优惠" and "家庭单优惠" not in u.get("topics", []):
            u["topics"] = u.get("topics", []) + ["家庭单优惠"]
        def done(text: str, route_: str, ev: list[Chunk] | None = None, nums: set | None = None) -> Answer:
            # program text is written to length already: no voice cap (it cut the discount sentence off)
            return Answer(text, route=route_, understanding=u, evidence=ev or [], structured=nums or set(),
                          timings={**timings, "total": time.perf_counter() - t0})

        gap_tail = "".join(GAP_FALLBACK[g] for g in u.get("gaps", []))
        # The understanding module also labels "ICU补贴是多少""VIP病房能报吗" as 报价; answering those with a price is
        # wrong (第八轮验收 5 calls). The price template is used only when the words ask for a price or only give conditions.
        price_asked = (bool(_PRICE_Q.search(question)) or bool({"age", "has_social"} & set(new_slots))
                       or not (_COVERAGE_WORDS.search(question) or QUESTION_MARKERS.search(question)))
        if (_ELIGIBLE_Q.search(question) and slots.get("age") is not None and not u.get("gaps")
                and not _COVERAGE_WORDS.search(question) and not _PRICE_Q.search(question)):
            return done(_eligibility_text(slots, question), "投保条件(模板)", topic_chunks("被保人年龄"),
                        {float(slots["age"]), float(MAX_AGE_NEW), 55.0, 30.0})
        if intent == "报价" and not price_asked:
            intent = u["intent"] = "保障" if (u.get("items") or _COVERAGE_WORDS.search(question)) else "其他"
        if intent == "报价":
            qt = _quote_text(slots)
            if qt is None:
                missing = "年龄" if slots.get("age") is None else "有没有社保"
                ask = "请问被保人今年多大年纪？" if missing == "年龄" else "请问您有社保吗？有没有社保价格不一样。"
                return Answer(ask, route="报价-追问", understanding=u, timings={**timings, "total": time.perf_counter() - t0})
            # The whole price answer is program text; the LLM's additions here were wrong (80 题 N037/N038/N042).
            text, nums = qt
            if _PRICE_CHANGE_Q.search(question) and (pc := _price_change_text(slots)):
                text, nums = pc[0], nums | pc[1]
            if (adds := addons_in(question)) and (aq := _addon_quote_text(slots, adds)):
                text, nums = text + aq[0], nums | aq[1]
            if _DISCOUNT_Q.search(question):
                text, nums = text + FAMILY_SAY, nums | {5.0, 10.0, 15.0, 20.0, 2.0, 3.0, 4.0}
            return done(text + gap_tail, "报价", topic_chunks("价格变化"), nums)
        if _WAIT_Q.search(question) and _clauses(question) <= 1 and not u.get("gaps"):
            return done(WAIT_TEXT, "等待期(模板)", topic_chunks("等待期"))
        if u.get("gaps") and set(u["gaps"]) <= TEMPLATE_ONLY_GAPS and _clauses(question) <= 1:
            lead = "这款产品的保险期间是一年。" if "续保" in u["gaps"] else ""
            return done(lead + gap_tail, "缺口(模板)", topic_chunks("保险期间"))

        for g in u.get("gaps", []):
            must_say.append(GAP_FALLBACK[g])
        # Underwriting questions: only the pre-existing-condition and disclosure clauses, so other
        # items' rules (e.g. an add-on's 20% for nodules) are not dragged into the answer.
        underwriting_only = intent == "缺口" and "疾病核保" in u.get("gaps", [])
        if underwriting_only:
            u["items"], u["topics"] = [], ["既往症", "如实告知"]
        for item in u.get("items", []):
            evidence += item_chunks(item)
        for t in u.get("topics", []):
            if t == "家庭单优惠":
                evidence.append(Chunk("family", "费率表", FAMILY_RULE, "费率表 家庭单用户优享价格规则说明"))
            else:
                evidence += topic_chunks(t)
        if self.retriever is not None and not underwriting_only:
            # Retrieval uses the user's own words and runs alongside understanding (exp13: hit@5 90.2% on the raw
            # question vs 83.2% on the understanding module's rewrite, which also had to be waited for).
            evidence += u["_retrieved"] if u.get("_retrieved") is not None else self.retrieve(question)
        # de-duplicate, keep order
        seen, ev = set(), []
        for c in evidence:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                ev.append(c)
        evidence = ev

        if agent_steps > 0:
            from agent.evidence_agent import gather

            t1 = time.perf_counter()
            extra, steps = gather(question, slots, known.get("_history", ""), evidence, client=self.client,
                                  max_steps=agent_steps, budget_s=agent_budget_s, retriever=self.retriever)
            seen = {c.chunk_id for c in evidence}
            evidence += [c for c in extra if c.chunk_id not in seen]
            timings["agent"] = time.perf_counter() - t1
            timings["agent_steps"] = steps

        ctx = grouped_context(evidence)
        shown = {k: v for k, v in slots.items() if not str(k).startswith("_")}
        history = known.get("_history", "")
        user = (f"已知用户条件：{json.dumps(shown, ensure_ascii=False)}\n"
                + (f"【最近对话】\n{history}\n" if history else "")
                + f"【报价结果】{must_say[0] if intent == '报价' and must_say else '无'}\n"
                f"【必须原样包含的话】{'；'.join(must_say[1:] if intent == '报价' else must_say) or '无'}\n"
                f"【资料】\n{ctx}\n\n用户：{question}")
        msgs = [{"role": "system", "content": _ANSWER_SYS + (VOICE_RULES if voice else "")}, {"role": "user", "content": user}]

        text, violations, attempts = "", [], 0
        for attempts in (1, 2):
            t1 = time.perf_counter()
            r = cloud.chat_stream(msgs, model=self.slow_model, client=self.client, temperature=0.3)
            timings.setdefault("generate_ttft", r.ttft_s)
            timings[f"generate_{attempts}"] = time.perf_counter() - t1
            text = r.text.strip()
            # Checked against the routed gaps only: checking against either path's gaps (exp14 first run) turned
            # coverage answers such as "癌症住院能报" into violations whenever one path misread the question.
            violations = _guard(question, text, evidence, structured, u.get("gaps"), gaps_judged=u.get("_gaps_judged", False),
                                fact_check=self.fact_check)
            if not violations:
                break
            # Rewrite as a fresh generation with the problems as constraints: sending the old answer back as a
            # dialogue made the model apologise to the customer ("抱歉，刚才说得不够准确", 第七轮).
            msgs = [msgs[0], {"role": "user", "content": user + "\n\n【注意】回答时避免以下问题："
                              + "；".join(violations) + "。"}]
        fallback = False
        if violations:
            fallback = True
            if not must_say and any("社保抵扣免赔额" in v for v in violations):
                must_say = [GAP_FALLBACK["社保抵扣免赔额"]]
            text = "".join(must_say) if must_say else (_item_template(u.get("items", [])) or "这个问题我需要帮您转专员确认一下。")
        text = addon_notice(text, evidence)
        if voice:
            text = cap_for_voice(text, must_say)
        timings["total"] = time.perf_counter() - t0
        return Answer(text, route=route, understanding=u, evidence=evidence, structured=structured,
                      violations=violations, attempts=attempts, fallback_used=fallback, timings=timings)


# --- streaming variant used by the voice loop --------------------------------------------------------

VOICE_MAX_CHARS = 50
FOLLOW_UP = "要不要我接着说说？"


def cap_for_voice(text: str, keep: list[str], limit: int = VOICE_MAX_CHARS) -> str:
    """Keep whole sentences up to ``limit`` characters (always the first one and any sentence that
    contains a required phrase); if anything was cut, end with a follow-up question."""
    from agent.speech import SentenceFeeder

    f = SentenceFeeder(None)
    sents = f.feed(text) + f.flush()
    out, n, cut = [], 0, False
    for i, s in enumerate(sents):
        required = any(k and k in s for k in keep)
        if i == 0 or required or (not cut and n + len(s) <= limit):
            out.append(s)
            n += len(s)
        else:
            cut = True  # keep a prefix: once a sentence is skipped, later ones are skipped too
    if cut and not any(x.endswith(("？", "?")) for x in out[-1:]):
        out.append(FOLLOW_UP)
    return "".join(out)


VOICE_RULES = ("\n8. 这是电话通话：一次只讲最关键的一到两点，整段不超过50个字；"
               "如果还有重要内容没讲，最后用一句话问用户要不要继续了解。不要使用“首先、其次、最后”这类连接词。")


def stream_answer(pipe: "Pipeline", question: str, known: dict | None = None, voice: bool = True):
    """Yield (event, payload, t) as the answer is produced, guarding each sentence before release.

    Events: "understood", "sentence" (a guarded, speakable sentence), "dropped" (a sentence that
    failed a guard), "done". Times are seconds since the call started.
    """
    from agent.speech import SentenceFeeder

    t0 = time.perf_counter()
    known = dict(known or {})
    if is_identity_question(question):
        yield "sentence", f"{IDENTITY_REPLY}价格和保障我都能帮您讲清楚，您想先了解哪方面？", time.perf_counter() - t0
        yield "done", None, time.perf_counter() - t0
        return
    u = understand(question, known, client=pipe.client, model=pipe.fast_model)
    yield "understood", u, time.perf_counter() - t0
    slots = {**known, **{k: v for k, v in (u.get("slots") or {}).items() if v is not None}}
    intent = u.get("intent", "其他")
    if intent == "身份":
        yield "sentence", f"{IDENTITY_REPLY}价格和保障我都能帮您讲清楚，您想先了解哪方面？", time.perf_counter() - t0
        yield "done", None, time.perf_counter() - t0
        return
    evidence: list[Chunk] = []
    structured = {float(v) for v in slots.values() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    must_say: list[str] = []
    if intent == "报价":
        qt = _quote_text(slots)
        if qt is None:
            ask = "请问被保人今年多大年纪？" if slots.get("age") is None else "请问您有社保吗？有没有社保价格不一样。"
            yield "sentence", ask, time.perf_counter() - t0
            yield "done", None, time.perf_counter() - t0
            return
        text, nums = qt
        # The quote is deterministic: speak it right away, before the LLM adds anything.
        yield "sentence", text, time.perf_counter() - t0
        structured |= nums
        evidence += topic_chunks("价格变化")
        must_say = []
        quote_said = text
    else:
        quote_said = ""
    for g in u.get("gaps", []):
        must_say.append(GAP_FALLBACK[g])
    for item in u.get("items", []):
        evidence += item_chunks(item)
    for t in u.get("topics", []):
        evidence += [Chunk("family", "费率表", FAMILY_RULE, "费率表")] if t == "家庭单优惠" else topic_chunks(t)
    if pipe.retriever is not None:
        evidence += pipe.retriever.search(u.get("rewrite") or question, k=4)
    seen, ev = set(), []
    for c in evidence:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            ev.append(c)
    ctx = "\n".join(f"[{i + 1}] {c.text}（出处：{c.source}）" for i, c in enumerate(ev))
    user = (f"已知用户条件：{json.dumps(slots, ensure_ascii=False)}\n"
            f"【报价结果】{quote_said or '无'}（已经说过，不要重复）\n"
            f"【必须原样包含的话】{'；'.join(must_say) or '无'}\n【资料】\n{ctx}\n\n用户：{question}")
    msgs = [{"role": "system", "content": _ANSWER_SYS + (VOICE_RULES if voice else "")}, {"role": "user", "content": user}]

    # Guard the whole (short) answer before speaking: dropping single sentences from a streamed
    # answer leaves dangling connectives ("但……", "其次……"), see 实验 4.
    text, violations = "", []
    for attempt in (1, 2):
        text = cloud.chat(msgs, model=pipe.slow_model, client=pipe.client, temperature=0.3).strip()
        if quote_said:  # the quote was already spoken; drop sentences that repeat its amounts
            nums = {n.value for n in extract_numbers(quote_said)}
            text = "".join(s for s in SentenceFeeder(None).feed(text + "\n")
                           if not ({n.value for n in extract_numbers(s)} & nums))
        violations = _guard(question, text, ev, structured, u.get("gaps"))
        if not violations:
            break
        yield "dropped", {"text": text, "violations": violations}, time.perf_counter() - t0
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": f"上面的回答有问题：{'；'.join(violations)}。请只用资料里的内容重写。"}]
    if violations:
        text = "".join(must_say) or _item_template(u.get("items", [])) or "这个问题我帮您转专员确认一下。"
    if voice:
        text = cap_for_voice(text, must_say, VOICE_MAX_CHARS - (len(quote_said) if quote_said else 0))
    feeder = SentenceFeeder()
    for piece in feeder.feed(text) + feeder.flush():
        yield "sentence", piece, time.perf_counter() - t0
    yield "done", None, time.perf_counter() - t0


# --- baseline: plain RAG ---------------------------------------------------------------------------

_BASELINE_SYS = "你是保险客服。根据资料回答用户关于尊享e生·百万医疗保险2026版的问题，口语化，不超过100字。"


def rate_chunks() -> list[Chunk]:
    """Rate tables as text, so plain RAG can at least see prices."""
    from agent.knowledge import rate_table

    out = []
    for pay in ("年缴", "月缴"):
        t = rate_table(f"{pay}_必选")
        for lo, hi in t.bands:
            row = t.cells[(lo, hi)]
            out.append(Chunk(f"rate:{pay}:{lo}", "费率表",
                             f"{pay}必选计划 {lo}到{hi}岁：有社保{row['必选计划_有社保']:g}元，无社保{row['必选计划_无社保']:g}元",
                             f"{pay}费率表 {lo}-{hi}岁"))
    return out


def baseline_answer(question: str, retriever: HybridRetriever, client=None, model: str | None = None, k: int = 6) -> Answer:
    t0 = time.perf_counter()
    ev = retriever.search(question, k=k)
    ctx = "\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(ev))
    r = cloud.chat_stream([{"role": "system", "content": _BASELINE_SYS},
                           {"role": "user", "content": f"资料：\n{ctx}\n\n用户：{question}"}],
                          model=model or cloud.SLOW_MODEL, client=client, temperature=0.3)
    return Answer(r.text.strip(), route="baseline", evidence=ev, timings={"total": time.perf_counter() - t0, "generate_ttft": r.ttft_s})
