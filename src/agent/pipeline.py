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
    other_conclusions, renewal_conclusion, social_deduction_conclusion,
)
from agent.knowledge import Chunk, clause_chunks, coverage_items
from agent.facts import SHORT, item_facts
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


_MERGED_EXTRA = """
另外输出两个字段（与上面的字段放在同一个 JSON 里）：
 "act": 用户这句话在销售对话中的作用，从 %(acts)s 中选一个，
 "gaps_now": 只看用户最新这一句（不考虑之前的对话），是否在问资料里没有的事项，从 ["疾病核保","续保","社保抵扣免赔额","条款清单"] 中选，没有则为空列表。
act 的含义：提问：问保障、价格、规则等；提供信息：回答坐席的问题或补充、纠正自己的情况（年龄、社保、给谁买、病史）；
附和：嗯、好的、你说；同意投保：愿意买、要链接、怎么买、发给我；拒绝：不需要、没兴趣、别打了、不买；
异议：嫌贵、要考虑、已经有医保或别的保险、不信任；要人工：要求转人工或真人；投诉：表达不满、要投诉；
身份：问是不是真人、机器人、AI、录音。一句话里既有信息又有提问时选"提问"。
gaps_now 的含义：
- "疾病核保"：用户说自己或家人有某种病、做过手术、在吃药，问能不能买、能不能承保、会不会影响投保、买了能不能报这个病；
- "续保"：问能不能续保、保证续保、停售后能否续、老了能不能续、明年还能不能买；
- "社保抵扣免赔额"：问医保/社保报销的部分能不能抵扣或计入免赔额，医保报完剩下的怎么报、还能不能报；
- "条款清单"：问重疾清单、特药清单、指定医院名单等具体清单。
问保障范围、价格、等待期、免赔额是多少，都不算。"""


def understand_merged(question: str, known: dict, client=None, model: str | None = None) -> dict:
    """One call instead of three (understand + classify_act + detect_gaps), 第十一轮对比实验。

    Returns the ``understand`` dict plus ``act`` and ``gaps_now`` (gaps judged on this sentence only)."""
    from agent.small_model import ACTS

    sys = (_UNDERSTAND_SYS % {"intents": INTENTS, "topics": TOPIC_NAMES, "gaps": GAP_NAMES,
                              "catalog": _catalog(), "max_items": MAX_ITEMS}
           + _MERGED_EXTRA % {"acts": ACTS})
    history = known.get("_history")
    shown = {k: v for k, v in known.items() if not k.startswith("_")}
    user = (f"已知条件：{json.dumps(shown, ensure_ascii=False)}\n"
            + (f"【最近对话】\n{history}\n" if history else "")
            + f"用户：{question}")
    raw = cloud.chat([{"role": "system", "content": sys}, {"role": "user", "content": user}],
                     model=model or cloud.FAST_MODEL, client=client, temperature=0,
                     response_format={"type": "json_object"})
    try:
        u = json.loads(raw)
    except json.JSONDecodeError:
        u = {"intent": "其他", "slots": {}, "items": [], "topics": [], "gaps": [], "rewrite": question}
    u.setdefault("slots", {})
    u["items"] = [i for i in u.get("items", []) if i in COVERAGE][:MAX_ITEMS]
    u["topics"] = [t for t in u.get("topics", []) if t in TOPIC_NAMES]
    u["gaps"] = [g for g in u.get("gaps", []) if g in GAP_NAMES]
    u["act"] = u.get("act") if u.get("act") in ACTS else "提问"
    u["gaps_now"] = [g for g in (u.get("gaps_now") or []) if g in ("疾病核保", "续保", "社保抵扣免赔额", "条款清单")]
    return u


# --- generation ----------------------------------------------------------------------------------

_ANSWER_SYS = """你是众安保险的 AI 电销坐席，正在电话里给用户介绍尊享e生·百万医疗保险2026版。
要求：
1. 只根据【资料】回答。资料里没有的内容不要说，不要推测，不要用常识补充。
2. 回答里的每个数字（金额、比例、天数、年龄）都必须在【资料】或【报价结果】里原样出现。价格只能用【报价结果】里的数字。
3. 口语化，先说结论；逐一回答用户问到的事项，只讲回答必需的事实和条件，不为凑字数重复或补充无关信息；不要 Markdown、不要列表符号、不要编号。
4. 讲保障时只补与所问直接相关的关键限制（如免赔额、等待期、医院范围或既往症），不要附加一整套通用提醒。
5. 禁止说"什么病都能报""保证续保""肯定能赔"之类的承诺；涉及病史要提醒如实告知，能否投保以核保结果为准，不要替核保下结论。
6. 如果【必须原样包含的话】不为空，把它自然地放进回答里。
7. 如果相关责任属于加油包（资料里写着"××加油包"），要说明这是需要另外选购的加油包，不在基础版里。
8. 资料按【组】给出，每组只属于一项责任或一份文件。每句话只能依据同一组资料，不同责任的规则（免赔额、共享范围、比例、天数）不能互相套用。
9. 说赔付比例时，如果资料里还有"未以社保身份就诊结算"的比例，要一并说明。注意：这个降低后的比例只适用于"按有社保身份投保、但看病没用社保结算"的情况；按无社保投保的人，扣除免赔额后仍按资料中的正常比例赔付。
10. 资料里没有理赔流程、理赔材料、理赔时效、APP 或线上操作方式，被问到就说以条款为准、可以转专员，不要描述流程。
11. 不要说"抱歉""我重新说""刚才说错了"之类的话，直接回答。
14. ICU（重症监护室）的床位费属于住院医疗费用，不要说ICU的医疗费基础版不报；ICU津贴是住院津贴加油包里按天另给的补贴，两者不同。
15. 用户一句话问了两件事，两件都要回答，每件一句。
16. 用户问每天给多少、一次或一年最多报多少时，直接说出资料里的具体数字，不要只说"需要另外选购"。
17. "社保报销后剩余部分按100%赔"只适用于特定药品里社保目录内的药；一般医疗、门急诊是扣除1万元免赔额后按100%赔。
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
           fact_check: bool | str = "multi", gaps_judged: bool = False, features: frozenset = frozenset(),
           client=None) -> list[str]:
    groups = [c.meta.get("item") or c.doc for c in evidence] if "D" in features else None
    v = number_guard(text, [c.text for c in evidence], structured, groups).violations
    # When the caller already judged gaps (CallSession: current-sentence detector), only those gaps are
    # enforced; the keyword rule alone misfires on e.g. "得了癌症……能报吗" (第六轮重测).
    v += (gap_guard("", text, gaps) if gaps_judged else gap_guard(question, text, gaps)).violations
    # "医保报完剩下的扣1万再报" is a conclusion on the 社保抵扣 gap whatever the question was (第六轮 2 calls)
    v += social_deduction_conclusion(text).violations
    v += renewal_conclusion(text).violations
    v += other_conclusions(text).violations
    v += compliance_guard(text).violations
    if re.search(r"基础版(?:不|没有|不予)(?:报销|保障|赔付)?住院(?:费|费用|保障|责任)", text):
        v.append("基础版包含一般医疗住院保障，不可说没有住院责任")
    from agent import small_model

    items = {c.meta.get("item") for c in evidence if c.meta.get("item")}
    run_fact = fact_check is True or fact_check == "always" or (fact_check == "multi" and len(items) >= 2)
    jobs = {}
    with ThreadPoolExecutor(3) as ex:  # the model checks are independent: run them side by side
        if gaps:
            jobs["gap"] = ex.submit(small_model.gap_review, question, text)
        if run_fact and evidence and not v:
            jobs["fact"] = ex.submit(small_model.fact_check, text, grouped_context(evidence))
        if "B" in features and items and not v:
            from agent.facts import check_claims

            jobs["claims"] = ex.submit(check_claims, text, client)
    if "gap" in jobs and jobs["gap"].result():
        v.append("模型复核：对缺口问题给出了结论")
    if "fact" in jobs:
        v += [f"事实核对：{p}" for p in jobs["fact"].result()]
    if "claims" in jobs:
        v += [f"规则说错：{p}" for p in jobs["claims"].result()]
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
_ELIGIBLE_Q = re.compile(r"(还)?(能|可以|能不能)(买|投|投保)|多大(年纪|岁数)?(能|可以)?(买|投)|(有没有|有)(什么)?(年龄)?限制|"
                         r"(年纪|岁数|年龄).{0,6}限制")


_HEALTH_RE = re.compile(r"病|症|炎|瘤|癌|结节|囊肿|息肉|高血压|糖尿病|三高|乙肝|手术|吃药|住过院|血压|血糖|血脂")


def age_stated(text: str, age) -> bool:
    """The age must appear in the words as an age: a standalone number (not part of "636"), not an amount ("63元")."""
    from agent.guardrail import cn_to_int

    a = int(age)
    for m in re.finditer(r"(?<!\d)(\d{1,3})(?!\d)\s*(元|块|万|%|天|个月)?", text):
        if int(m.group(1)) == a and not m.group(2):
            return True
    return any(cn_to_int(m.group(1)) == a for m in re.finditer(r"([一二两三四五六七八九十百]{1,4})(周)?(岁|了)", text))


def _eligibility_text(slots: dict, question: str) -> str:
    age = int(slots["age"])
    if age > MAX_AGE_NEW:
        return f"{age}岁超过了首次投保的年龄上限{MAX_AGE_NEW}周岁，这款产品买不了。"
    asks_outpatient = bool(re.search(r"门急诊|门诊", question))
    if asks_outpatient:
        return (f"门急诊加油包限55周岁及以下，{age}岁{'符合年龄要求' if age <= 55 else '不能选'}。"
                + (f"其余加油包的首次投保年龄上限为{MAX_AGE_NEW}周岁，您符合年龄要求，仍须按各包条件及核保确认。"
                   if re.search(r"哪些|其他|其余", question) else "")
                + ("能否承保以健康告知和核保为准。" if age <= 55 else ""))
    if re.search(r"多大|几岁|年龄限制|年龄范围", question):
        return f"首次投保限出生满30天到{MAX_AGE_NEW}周岁，{age}岁符合年龄要求；能否承保以核保为准。"
    return f"{age}岁符合首次投保年龄要求；能否承保以健康告知和核保为准。"


_DISCOUNT_Q = re.compile(r"优惠|打折|折扣|便宜|全家|家里人|一家|一起买")
_TOTAL_Q = re.compile(r"加起来|一共|总共|合计|都加上|全加上")
_PEOPLE = [(5, r"五口|五个人|5个人|一大家子"), (4, r"四口|四个人|4个人"), (3, r"三口|三个人|3个人"),
           (2, r"两个人|2个人|两口子|俩|夫妻|我们两个|我和(我)?(老公|老婆|爱人|媳妇|对象|丈夫|妻子)")]


def people_count(text: str) -> int | None:
    return next((n for n, pat in _PEOPLE if re.search(pat, text)), None)


def _family_text(slots: dict, n: int, question: str) -> tuple[str, set] | None:
    """Family-order discount on this insured's price; "都39岁、都没社保" gives the total for everyone."""
    from agent.quote import family_discount

    age, social = slots.get("age"), slots.get("has_social")
    if age is None or social is None or int(age) > MAX_AGE_NEW:
        return None
    y = quote(Insured(int(age), bool(social)), "年缴").total
    d = family_discount(n)
    per = round(y * (1 - d), 2)
    same = bool(re.search(r"都", question))
    txt = (f"{n}个人一起买是家庭单，每人优惠{d * 100:g}%，{_who(slots) if not same else '每人'}一年{per:g}元"
           + (f"，{n}个人一共{round(per * n, 2):g}元" if same else "") + "，折后金额以投保页面为准。")
    return txt, {float(n), round(d * 100, 2), per, round(per * n, 2), y}


def _total_text(slots: dict, addons: list[str]) -> tuple[str, set] | None:
    age, social = slots.get("age"), slots.get("has_social")
    if age is None or social is None:
        return None
    r = quote(Insured(int(age), bool(social), gender=slots.get("gender"), addons=addons), "年缴")
    if not r.lines or r.errors or r.needs:
        return None
    parts = "、".join(l.label.split(" ", 1)[1].replace("必选计划", "基础版") + f"{l.amount:g}元" for l in r.lines)
    txt = f"{_who(slots)}{int(age)}岁，{parts}，加起来一年一共{r.total:g}元。"
    return txt, {float(int(age)), r.total, *(l.amount for l in r.lines)}
FAMILY_SAY = "优惠的话，给家人一起买有家庭单优惠：2个人每人优惠5%，3个人10%，4个人15%，5个人及以上20%。"
_PRICE_CHANGE_Q = re.compile(r"以后|每年|明年|后年|将来|涨|一直(都)?(是)?这个价|固定|变不变|会变|老了")
_WAIT_Q = re.compile(r"等待期|等多久|等多长|多久(才)?(能|可以)?(用|生效|报|赔)|刚买就|买完就|马上就?(生病|住院)|意外.{0,10}(等|等待)|要等")
WAIT_TEXT = "从保单生效起，疾病等待期30天；意外住院和门急诊不受此限制。"
WAIT_ADDON_TEXT = "药省保加油包等待期7天。"
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
    # Generic names such as "门急诊医疗" and "住院医疗费用" also occur in base-plan
    # answers. A retrieval hit alone must not add an unrelated add-on warning.
    specific = {
        "R09": r"家庭共享免赔额|共享免赔额加油包",
        "R10": r"重疾关爱|重疾给付",
        "R11": r"扩展特需医疗|特需医疗加油包",
        "R12": r"住院医疗费用补偿|住院补偿加油包",
        "R13": r"门急诊医疗加油包|门诊加油包|门急诊加油包",
        "R14": r"住院津贴|一般住院津贴",
        "R15": r"ICU津贴|重症监护病房住院津贴",
        "R16": r"药省保|互联网药品费用医疗保险金",
    }
    for c in evidence:
        iid = c.meta.get("item")
        if not iid or COVERAGE[iid]["责任&加油包"] == "必选责任":
            continue
        it = COVERAGE[iid]
        pack = it["责任&加油包"]
        if (pack in out or pack.replace("尊享", "") in out or re.search(specific.get(iid, r"(?!)"), out)) and not re.search(r"另外|额外|单独|加购|另购|另买|附加|自选", out):
            out = out.rstrip() + f"{pack.replace('尊享', '')}需要在基础版之外另外选购。"
            break
    return out


# --- 方案 A / D helpers ------------------------------------------------------------------------------

FACTS_RULE = """
13. 本题用"只选不写"的方式回答：不要直接写回答，只输出 JSON：
{"opening": "开头一句，直接回应用户，不含任何数字和保障规则", "facts": [要讲的【事实】编号，按讲的顺序，最多4个],
 "calc": "如果用户问能报多少钱、能拿多少，写一句算出的结果（只能用【事实】里的数和用户说的数）；否则为空",
 "closing": "可选的一句补充（如免责、提醒），不含保额、免赔额、比例、天数这些规则；没有就为空"}
保额、免赔额、赔付比例、等待期、每天给多少、是否要另外买、年龄限制，只能通过选【事实】编号来说。"""


_RULE_WORDS = re.compile(r"\d|[一二三四五六七八九十百千]+(万|元|天|%)|免赔|比例|保额|另外选购|加油包|等待期|最高|每天|万元|%")


def assemble_facts(raw: str, facts: list[str]) -> str:
    """方案 A: the program writes the answer from the model's plan; rule sentences come only from the fact list."""
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    def free(t) -> str:  # free text may not carry rules: a sentence with a number or a rule word is dropped
        t = str(t or "").strip("。 ")
        return "" if _RULE_WORDS.search(t) else t

    parts = [free(d.get("opening"))]
    for k in d.get("facts") or []:
        try:
            k = int(str(k).lstrip("Ff"))
        except ValueError:
            continue
        if 1 <= k <= len(facts):
            parts.append(facts[k - 1])
    parts += [str(d.get("calc") or "").strip("。 "), free(d.get("closing"))]
    return "。".join(dict.fromkeys(p for p in parts if p)) + "。"


def assemble_voice_slots(raw: str, expected: int) -> str:
    """Experimental opt-in: release only a complete one-to-one set of voice answer slots."""
    try:
        answers = json.loads(raw).get("answers")
    except (ValueError, AttributeError):
        return ""
    if not isinstance(answers, list) or len(answers) != expected:
        return ""
    sentences = []
    for i, answer in enumerate(answers, 1):
        if not isinstance(answer, dict) or answer.get("id") != i:
            return ""
        conclusion = answer.get("conclusion")
        if not isinstance(conclusion, str) or not conclusion.strip():
            return ""
        fields = [answer.get(key, "") for key in ("conclusion", "numbers", "condition")]
        if not all(isinstance(value, str) for value in fields):
            return ""
        sentence = "，".join(value.strip(" ，;；。") for value in fields if value.strip(" ，;；。"))
        sentences.append(sentence + "。")
    return "".join(sentences)


def scoped_rule_reply(question: str, u: dict) -> tuple[str, list[str], list[str]] | None:
    """Compose frequently paired obligations from curated responsibility rules."""
    obligations = u.get("obligations", [])
    # Additional questions must stay on the general route rather than be swallowed by a template.
    if not obligations or len(obligations) > 3:
        return None
    if ("住院补偿加油包" in question or "住院医疗费用补偿加油包" in question):
        if ("基础版" in question and re.search(r"一万以内|1万以内|免赔额以内", question)
                and all(re.search(r"住院|结节|赔|报", q) for q in obligations)
                and not re.search(r"门诊|医保|社保|已经|累计|续保|多少钱", question)):
            return ("基础版一般住院每年1万免赔；另购住院补偿加油包，合规住院费用按100%赔、全年最多1万，结节等约定疾病按20%赔；既往症及相关费用不赔。",
                    ["R01", "R12"], ["既往症"])
    if "疾病核保" in u.get("gaps", []) and "意外" in question and len(obligations) == 2:
        if all(re.search(r"报|赔|住院", q) for q in obligations):
            return ("投保前已有病及相关费用属于既往症免责；能否投保须健康告知，以核保为准。意外住院不受疾病等待期限制，符合一般医疗保障条件才可报。",
                    ["R01"], ["既往症", "如实告知", "等待期"])
    if ({"R01", "R02", "R15"} <= set(u.get("items", [])) and len(obligations) == 3
            and re.search(r"ICU|重症", question) and not re.search(r"花|天|最多|总额|算|私立|特需|VIP", question)
            and all(re.search(r"住院|门诊|门急诊|ICU|重症", q) for q in obligations)):
        return ("住院和一般门急诊共享每年1万免赔额，扣除后按100%赔；按有社保身份投保却没用社保结算，按60%赔。ICU津贴需另购，每天500元。",
                ["R01", "R02", "R15"], [])
    if (len(obligations) == 2 and re.search(r"意外|车祸", question) and re.search(r"生病|疾病", question)
            and all(re.search(r"等待|生病|疾病|车祸|意外", q) for q in obligations)
            and not re.search(r"多少|免赔|社保|医保|既往|高血压|加油包", question)):
        return ("意外住院不受疾病等待期限制，符合保障条件才可报；疾病住院有30天等待期，期间罹患的疾病不赔。",
                ["R01"], ["等待期"])
    return None


def obligation_guard(question: str, text: str, items: list[str], gaps: list[str]) -> list[str]:
    """Required scope conditions that a numeric whitelist cannot detect."""
    violations = []
    if {"R01", "R02"} <= set(items) and re.search(r"住院.*(?:门诊|门急诊)|门诊.*住院", question):
        if "共享" not in text and "共用" not in text:
            violations.append("一般医疗与一般门急诊共享年度免赔额和额度，不可各算一份")
    if "疾病核保" in gaps and not re.search(r"既往症|投保前.*(?:病|疾病)", text):
        violations.append("必须说明既往症及相关费用免责，不可只谈疾病等待期或核保")
    if "意外" in question and "疾病核保" in gaps and not re.search(r"意外.*(?:不受|没有|无).{0,8}等待期", text):
        violations.append("须回答意外部分，意外伤害不受疾病等待期限制")
    return violations


def hospital_calculation(question: str, slots: dict, items: list[str]) -> dict | None:
    """Compute a single R01 illustration only; do not infer settled/eligible expenses."""
    if "R01" not in items or not re.search(r"住院.*(?:花|费用|费|支出)|住院花", question):
        return None
    if re.search(r"重疾|癌|津贴|加油包|医保报|社保报|既往|核保|高血压|门诊|ICU|私立|特需|续保|已经|已抵|用掉|累计", question, re.I):
        return None
    amounts = [n.value for n in extract_numbers(question) if n.unit in ("", "元") and n.value >= 1000]
    if len(amounts) != 1:
        return None
    amount = amounts[0]
    balance = max(0, amount - 10000)
    lower = round(balance * .6, 2)
    return {"rule": "R01: max(eligible_expense-remaining_deductible,0)*ratio",
            "assumptions": ["全部费用符合一般医疗责任", "本年度1万元免赔额尚未抵扣", "未超过年度额度", "无其他补偿"],
            "numbers": [balance, lower, amount - balance, amount - lower],
            "text": f"假设费用全部符合一般医疗保障、年度1万免赔还未抵扣且无其他补偿，按100%赔{balance:g}元，自己付{amount-balance:g}元；若按有社保身份投保却没用社保结算，按60%赔{lower:g}元，自己付{amount-lower:g}元。实际还受合规费用、免赔余额和年度额度限制。"}


def derived_numbers(question: str, evidence: list[Chunk]) -> set[float]:
    """方案 D: one-step results from the user's numbers and the evidence's numbers ("住院5万" − 1万 = 4万)."""
    from agent.guardrail import extract_numbers as nums

    q = [n.value for n in nums(question) if n.value > 0]
    if len(q) >= 2:
        q.append(sum(q))
    ev = [n for c in evidence for n in nums(c.text)]
    amounts = {n.value for n in ev if n.unit in ("", "元") and n.value > 0}
    days = {n.value for n in ev if n.unit == "天"}
    pct = {n.value for n in ev if n.unit == "%"}
    out: set[float] = set()
    for a in q:
        for e in amounts | days:
            out |= {a - e, e - a, a + e}
            for p in pct:
                out |= {round((a - e) * p / 100, 2), round(a * p / 100, 2)}
        for d in days:
            out |= {(a - d) * e for e in amounts}
        out |= {a * e for e in amounts if e < 2000}  # days × daily amount
    return {round(x, 2) for x in out if x > 0}


_FIELD_WORDS = [("免赔", "免赔"), ("比例|赔多少|报多少|按多少", "赔"), ("多少|最高|额度|保额", "最"), ("等|等待", "等待期"),
                ("每天|一天|日", "每天"), ("加油包|另外|基础版", "选购")]


def spoken_fallback(item_ids: list[str], question: str) -> str:
    """方案 D: when an answer cannot be released, say only the asked item's relevant facts, in spoken sentences."""
    facts = item_facts(item_ids[0])
    want = [key for pat, key in _FIELD_WORDS if re.search(pat, question)]
    pick = [f for f in facts if any(k in f for k in want)] or facts[1:3]
    return "。".join(([facts[0]] if facts[0] not in pick else []) + pick[:3]) + "。"


def understand_split(question: str, known: dict, client=None, model: str | None = None, answer_planning: bool = True) -> dict:
    """方案 C: the understanding prompt split into three short ones run side by side (intent+gaps / items+topics / slots)."""
    from agent import small_model

    model = model or cloud.FAST_MODEL
    # 第十轮回归: one of the three calls occasionally hung ~12 s; each now gets 3 s and one retry, then a default
    fast = (client or cloud.llm_client()).with_options(timeout=3.0, max_retries=1)
    prev = known.get("_prev_utterance")
    history = known.get("_history")
    ctx = ((f"【最近对话】\n{history}\n" if history else "")
           + (f"【上一句】用户：{prev}（上一句意图：{known.get('_prev_intent')}）\n" if prev and not history else ""))

    def intent():
        raw = cloud.chat([{"role": "system", "content": _SPLIT_INTENT_SYS % {"gaps": GAP_NAMES}},
                          {"role": "user", "content": ctx + f"用户：{question}"}],
                         model=model, client=fast, temperature=0, response_format={"type": "json_object"})
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            d = {}
        return {"intent": d.get("intent") if d.get("intent") in INTENTS else "其他",
                "gaps": [g for g in d.get("gaps", []) if g in GAP_NAMES]}

    def items():
        raw = cloud.chat([{"role": "system", "content": _SPLIT_ITEMS_SYS % {"topics": TOPIC_NAMES, "catalog": _catalog()}},
                          {"role": "user", "content": f"用户：{question}"}],
                         model=model, client=fast, temperature=0, response_format={"type": "json_object"})
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            d = {}
        return {"obligations": [q.strip() for q in d.get("obligations", []) if isinstance(q, str) and q.strip()][:8],
                "items": [i for i in d.get("items", []) if i in COVERAGE][:MAX_ITEMS],
                "topics": [t for t in d.get("topics", []) if t in TOPIC_NAMES]}

    def safe(fn, default, *a):
        try:
            return fn(*a)
        except Exception:
            return default

    with ThreadPoolExecutor(3) as ex:
        fi = ex.submit(safe, intent, {"intent": "其他", "gaps": []})
        fit = ex.submit(safe, items, {"items": [], "topics": []}) if answer_planning else None
        fs = ex.submit(safe, small_model.extract_slots, {}, question, fast)
        u = {**fi.result(), **(fit.result() if fit else {"items": [], "topics": []}), "slots": fs.result(), "rewrite": question}
    if "疾病核保" in u["gaps"]:
        u["topics"] = list(dict.fromkeys(u["topics"] + ["既往症", "如实告知"]))
    return u


_SPLIT_INTENT_SYS = """判断保险电销通话中用户这句话的意图和是否问到资料里没有的事项，只输出 JSON：{"intent": ..., "gaps": [...]}
intent 之一：
- 报价：问价格、多少钱、每月/每年交多少、以后价格会不会变；只补充年龄、社保等条件而没问别的，也算报价
- 优惠：问优惠、打折、全家一起买便宜多少
- 保障：问能不能报、怎么报、报多少、免赔额、赔付比例、等待期、医院、津贴每天多少、加油包保什么
- 投保条件：问多大年纪能买、职业、什么时候生效、能买几份、能不能给家人买
- 免责：问某种病或某项费用本身在不在保障范围（没说自己已经有这个病）
- 缺口：问资料里没有的事项（见下面 gaps）
- 身份：问对方是不是真人、机器人、AI、录音
- 闲聊：寒暄、附和、说自己情况但没提问；其他：都不是
gaps 可选 %(gaps)s，规则：
- 疾病核保：用户说自己或家人已经有某种病、做过手术、在吃药，问能不能买、买了能不能报这个病。只是问"得了某病能不能报"（假设）不算。
- 续保：问能不能续保、保证续保、停售后怎么办。
- 社保抵扣免赔额：只在问医保/社保报销的钱能不能算进或抵扣1万免赔额时才算；问没走社保报多少、医保报过的会不会重复赔都不算。
- 条款清单：问重疾清单、特药清单、医院名单。
没有就输出空列表。"""

_SPLIT_ITEMS_SYS = """列出回答用户这句话需要查的保险责任和关键主题，只输出 JSON：{"items": [...], "topics": [...], "obligations": [...]}
obligations：逐项列出用户实际问到的待答问题（字符串），并列的保障、金额、条件要拆开；年龄、社保等背景条件不是待答问题。不要增加用户没问的事项，不要写答案。核保不确定也须保留其他能回答的保障问题。
items 从下面责任目录中选编号，问到几项就列几项（最多 5 个），只列直接相关的；没问保障就为空。
- 问免赔额以内、自己先掏的部分能不能报 → 含 R01、R12、R09；问确诊重病能不能直接拿钱 → 含 R10、R03；
- 问门诊/急诊 → R02（基础版）；问门诊加油包 → R13；问住院津贴/ICU补贴 → R14、R15；问网上买药 → R16。
topics 从 %(topics)s 中只选用户实际问到或直接影响答案的主题；不要复制整个目录。
责任目录：
%(catalog)s"""


def combine_gaps(understood: list[str], detected: list[str], question: str = "") -> list[str]:
    """Gaps the answer is routed on. A gap counts only when the understanding module and the gap detector
    agree (exp12: each alone misfires on different questions, e.g. "确诊癌症能直接拿到钱吗"); the detector
    names the type. Two exceptions, where one path is enough: clause lists on the understanding module (it does
    not misfire on them), and 社保抵扣免赔额, whose fallback is true either way (it states the 补偿原则)."""
    out = list(detected) if understood else []
    # a third vote: the keyword rule. Two of three (understanding, detector, rule) is enough (第十轮回归: with the split
    # understanding, "我现在有高血压，能买你这个保险不" was missed by one path and the answer said "50岁能投保")
    rule = gap_of(question) if question else None
    if rule and rule.name in understood + detected and rule.name not in out:
        out.append(rule.name)
    # one path is enough for 社保抵扣 only when the words are about the deductible ("特药里社保目录内的药怎么报" is not)
    question_mentions_social = not question or bool(re.search(r"(医保|社保)[^。？?]{0,12}(免赔|抵|算进|算在|计入|报完|报了|报过|剩下|剩余)|"
                                                              r"(免赔|一万|1万)[^。？?]{0,12}(医保|社保)", question))  # "既往症是什么意思" misfired (第八轮回归)
    asks_renewal = not question or bool(re.search(r"续|停售|下架|到期|不卖了", question))  # 第十轮回归: split prompt missed 续保
    if not asks_renewal:
        out = [g for g in out if g != "续保"]
    out = [g for g in out if g != "条款清单" or not question or re.search(r"清单|名单|哪些药|哪[几些]家医院|列出", question)]
    if question and not re.search(r"(医保|社保).{0,12}(报销|报了|报过|报完|抵|计入|算进|算在)|(抵扣|计入|算进|算在).{0,8}(医保|社保)", question):
        out = [g for g in out if g != "社保抵扣免赔额"]
        question_mentions_social = False
    for g, either in (("条款清单", "条款清单" in understood and (not question or bool(re.search(r"清单|名单|哪些药|哪[几些]家医院|列出", question)))), ("社保抵扣免赔额", "社保抵扣免赔额" in understood + detected and question_mentions_social),
                      ("续保", "续保" in understood + detected and asks_renewal)):
        if either and g not in out:
            out.append(g)
    return out


def gaps_to_check(understood: list[str], detected: list[str]) -> list[str]:
    """Gaps the finished answer is checked against: either path is enough (checking costs little, missing costs a lot)."""
    return list(dict.fromkeys(understood + detected))


_DIRECT_SYS = """你是众安保险AI电话助理，回答尊享e生百万医疗2026版问题，只依据本轮资料与程序报价，不编造。
直接理解原用户问题与最近对话，在内部选择本轮必须说的事实，再输出自然口语短答，不输出思考过程、JSON或问题清单。
事实必须直接回答用户实际问到的一件事，或省略后会令答案错误或误导。背景不是新问题，不自行扩写待答项。
共同条件只说一次，多个问题用短句并列。未问的保额、医院范围、等待期、流程、天数上限不主动介绍，答完即停，细节留给追问。
但用户问“能报多少”，每项被问责任的具体金额或额度必须回答；问每天、每次、每年，按该维度回答，不能拿另一责任数字代替。
金额题交付前在内部核对：用户点名的每项责任是否都有对应额度，不能只给比例或申请流程。仅问每月上限时只给每月金额，不加年度上限、比例或另购说明。
保留另购、具体责任范围、既往症免责和核保边界。一般规则不代表用户一定能承保或获赔；未知部分局部说明，已知部分仍回答。
赔付比例必须绑定该责任与投保身份：降低比例适用于按有社保身份投保却没用社保结算，不能只说没用社保就降低；投保身份未知须条件化。
一般医疗与一般门急诊共享年度免赔和额度；家庭共享免赔加油包不含一般门急诊。不得跨责任套用比例、等待期或限额。
简单题一两句，复合题每问一个短结论，通常争取70字，必要条件不可为字数删掉，不截断。不复述背景、不解释推导、不重复数字、不加销售引导或反问。
举例只学习取舍：问“投保后什么时候生效”，只答投保成功后次日零时生效，不主动讲疾病等待期；问某责任每月上限，只答月上限，不主动介绍年上限。
"""


def direct_topics(question: str) -> list[str]:
    patterns = {"等待期": r"等待|刚买|意外|车祸", "既往症": r"既往|投保前|高血压|结节|病史",
                "如实告知": r"核保|健康告知|我.{0,8}(有|患|得过).{0,8}(病|癌|血压)",
                "医院范围": r"医院|私立|特需|VIP", "社保结算比例": r"社保|医保",
                "家庭单优惠": r"家庭|一家|几口|[2345四三两五]口|优惠", "总保额": r"总保额|总共保|保什么",
                "其他免责": r"美容|怀孕|牙科|近视|抑郁", "补偿原则": r"报销过|报完|重复赔"}
    return [topic for topic, pattern in patterns.items() if re.search(pattern, question, re.I)]


class Pipeline:
    def __init__(self, retriever: HybridRetriever | None, client=None, slow_model: str | None = None,
                 fast_model: str | None = None, fact_check: str = "off", features: frozenset | set = frozenset({"C", "D"}),
                 direct_answer: bool = False):
        self.direct_answer = direct_answer
        self.fact_check = fact_check  # "always" | "multi" (only when evidence spans 2+ coverage items) | "off"
        # 第十轮消融的方案开关（08 方案 A–D）：A 只选不写、B 三元组比对、C 理解拆分、D 数字护栏算术与口语兜底。
        # 默认 C+D（exp15：qa_v6 87.1%/90.3%，基线 77.4%/79.0%）；A、B 消融未通过，保留为可选
        self.features = frozenset(features)
        self.retriever = retriever
        self.client = client or cloud.llm_client()
        self.slow_model = slow_model or cloud.SLOW_MODEL
        self.fast_model = fast_model or cloud.FAST_MODEL
        self._responsibility_retriever = Retriever([item_chunks(iid)[0] for iid in COVERAGE]) if direct_answer else None

    def understand(self, question: str, known: dict) -> dict:
        u = (understand_split(question, known, self.client, self.fast_model, answer_planning=not self.direct_answer) if self.direct_answer or "C" in self.features
             else understand(question, known, self.client, self.fast_model))
        slots = u.get("slots") or {}
        if slots.get("age") is not None and not age_stated(question, slots["age"]):
            slots = {k: v for k, v in slots.items() if k != "age"}  # "636一年是吧" read as age 63 (第十轮回归)
        u["slots"] = slots
        return u

    def _answer_rest(self, question: str, known: dict, slots: dict, voice: bool, handled: re.Pattern) -> str:
        """The part of a two-question sentence that a template did not answer ("我30岁有社保多少钱，住院怎么报？")."""
        if known.get("_rest_depth"):
            return ""
        parts = [c for c in re.split(r"(?<=[，,。？?；;！!])", question) if c.strip()]
        rest = "".join(c for c in parts if not handled.search(c) and _COVERAGE_WORDS.search(c))
        if not rest or len(parts) < 2:
            return ""
        sub_known = {**known, **{k: v for k, v in slots.items() if not str(k).startswith("_")}, "_rest_depth": 1}
        return self.answer(rest, sub_known, voice=voice).text

    def retrieve(self, question: str, k: int = 4) -> list[Chunk]:
        retrieved = self.retriever.search(question, k=k) if self.retriever is not None else []
        if not self.direct_answer:
            return retrieved
        # Responsibility routing uses the original question and local lexical retrieval, no LLM planner.
        ranked = self._responsibility_retriever.search(question, k=3, method="tfidf")
        aliases = {"R06": r"靶向药|特效药|特药|特定药|罕见病|基因检测", "R05": r"质子|重离子|光免疫",
                   "R08": r"护工", "R07": r"路费|住宿|异地转诊", "R09": r"共享免赔|免赔额.{0,4}共享",
                   "R13": r"门[急]?诊加油包|门急诊加油包", "R15": r"ICU|重症.*(补贴|津贴)"}
        exact = [item_chunks(iid)[0] for iid, pattern in aliases.items() if re.search(pattern, question, re.I)]
        return list({c.chunk_id: c for c in exact + retrieved + ranked}.values())

    def answer(self, question: str, known: dict | None = None, voice: bool = False,
               agent_steps: int = 0, agent_budget_s: float = 3.0, understanding: dict | None = None,
               on_sentence=None) -> Answer:
        """``known`` carries the call's profile; ``known["_history"]`` (recent turns, as text) and
        ``known["_prev_utterance"]`` come from CallSession. ``agent_steps`` > 0 lets the evidence
        agent add tool calls on top of the fixed routing, within ``agent_budget_s`` seconds.

        ``on_sentence``: if given, the answer is streamed — each sentence is guarded and handed to the
        callback as soon as the model finishes it, instead of after the whole answer (see ``_stream_generate``)."""
        known = dict(known or {})
        t0 = time.perf_counter()
        timings: dict = {}

        if is_identity_question(question):
            return Answer(f"{IDENTITY_REPLY}价格和保障我都能帮您讲清楚，您想先了解哪方面？", route="身份(规则)",
                          timings={"total": time.perf_counter() - t0})

        if understanding is None:
            from agent import small_model

            with ThreadPoolExecutor(3) as ex:
                fu = ex.submit(self.understand, question, known)
                fd = ex.submit(small_model.detect_gaps, question, self.client)
                fr = ex.submit(self.retrieve, question)
                u, detected = fu.result(), fd.result()
                u["_retrieved"] = fr.result()
            u["_gaps_check"] = gaps_to_check(u.get("gaps", []), detected)
            u["gaps"], u["_gaps_judged"] = combine_gaps(u.get("gaps", []), detected, question), True
            if u.get("intent") == "缺口" and not u["gaps"]:
                u["intent"] = "保障" if u.get("items") else "其他"
        else:
            u = {**understanding, "items": list(understanding.get("items", [])),
                 "topics": list(understanding.get("topics", []))}
        if self.direct_answer:
            u.pop("obligations", None)
            u.pop("_voice_questions", None)
            u["items"] = list(dict.fromkeys(c.meta["item"] for c in u.get("_retrieved", []) if c.meta.get("item")))[:MAX_ITEMS]
            u["topics"] = direct_topics(question)
            # Hypothetical coverage questions are not personal underwriting requests.
            if not re.search(r"(我|本人|爸|妈|父亲|母亲|家人|孩子|爱人).{0,16}(有|患|得过|得了|手术|吃药|病史)", question):
                u["gaps"] = [g for g in u.get("gaps", []) if g != "疾病核保"]
            if u.get("intent") == "缺口" and not u.get("gaps"):
                u["intent"] = "保障"
        # Explicitly asked base hospitalization must not disappear when an add-on is also mentioned.
        if re.search(r"基础版|一般医疗", question) and "住院" in question and "R01" not in u.get("items", []):
            u["items"] = ["R01"] + u.get("items", [])
        timings["understand"] = time.perf_counter() - t0
        trace = timings["trace"] = [{"stage": "flow" if self.direct_answer else "plan", "mode": "direct" if self.direct_answer else "slots",
                                     "question": question, "obligations": u.get("obligations", []),
                                     "items": list(u.get("items", [])), "gaps": list(u.get("gaps", []))}]
        # only what this sentence adds or changes: the model often echoes the known conditions back (第八轮回归 3 calls)
        new_slots = {k: v for k, v in (u.get("slots") or {}).items() if v is not None and known.get(k) != v}
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
            trace.append({"stage": "release", "source": "program", "route": route_, "chars": len(text)})
            return Answer(text, route=route_, understanding=u, evidence=ev or [], structured=nums or set(),
                          timings={**timings, "total": time.perf_counter() - t0})

        gap_tail = "".join(GAP_FALLBACK[g] for g in u.get("gaps", []))
        rule_reply = scoped_rule_reply(question, u)
        if rule_reply:
            text, item_ids, topics = rule_reply
            ev = [c for iid in item_ids for c in item_chunks(iid)] + [c for topic in topics for c in topic_chunks(topic)]
            trace.append({"stage": "resolve", "items": item_ids, "evidence": [c.chunk_id for c in ev],
                          "local_fallbacks": list(u.get("gaps", []))})
            trace.append({"stage": "compose", "source": "scoped_rules", "text": text})
            return done(text, "保障条件(模板)", ev)
        # The understanding module also labels "ICU补贴是多少""VIP病房能报吗" as 报价; answering those with a price is
        # wrong (第八轮验收 5 calls). The price template is used only when the words ask for a price or only give conditions.
        last_q = next((c for c in reversed(re.split(r"[。！？?，,；;]", question)) if QUESTION_MARKERS.search(c)), question)
        # judged on the question actually asked: "那比刚才贵了不少。这个免赔额是每年都1万吗？" asks about the deductible
        price_asked = ((bool(_PRICE_Q.search(last_q)) and not _COVERAGE_WORDS.search(last_q)) or bool({"age", "has_social"} & set(new_slots))
                       or not (_COVERAGE_WORDS.search(question) or QUESTION_MARKERS.search(question)))
        if (_ELIGIBLE_Q.search(question) and slots.get("age") is not None and not u.get("gaps")
                and not _COVERAGE_WORDS.search(question) and not _PRICE_Q.search(question)
                and not gap_of(question) and not _HEALTH_RE.search(question)):  # "我这高血压到底能不能买" is underwriting
            return done(_eligibility_text(slots, question) + self._answer_rest(question, known, slots, voice, _ELIGIBLE_Q),
                        "投保条件(模板)", topic_chunks("被保人年龄"), {float(slots["age"]), float(MAX_AGE_NEW), 55.0, 30.0})
        if intent == "优惠" and people_count(question) and slots.get("age") is not None and slots.get("has_social") is not None:
            intent, price_asked = "报价", True
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
            if re.search(r"(年|一年).*(月)|月.*年", question) and re.search(r"差|比较|划算|合算", question):
                y = quote(Insured(int(slots["age"]), bool(slots["has_social"]), gender=slots.get("gender")), "年缴")
                m = quote(Insured(int(slots["age"]), bool(slots["has_social"]), gender=slots.get("gender")), "月缴")
                total, difference = round(m.total * 12, 2), round(m.total * 12 - y.total, 2)
                text += f"月缴全年共{total:g}元，比年缴{'多' if difference >= 0 else '少'}{abs(difference):g}元。"
                nums |= {total, abs(difference)}
                trace.append({"stage": "calculate", "rule": "monthly*12-annual", "monthly": m.total,
                              "annual": y.total, "total": total, "difference": difference})
            if _PRICE_CHANGE_Q.search(question) and (pc := _price_change_text(slots)):
                text, nums = pc[0], nums | pc[1]
            if (adds := addons_in(question)) and (aq := _addon_quote_text(slots, adds)):
                text, nums = text + aq[0], nums | aq[1]
            if _TOTAL_Q.search(question):
                adds_all = list(dict.fromkeys(addons_in(question) + addons_in(known.get("_history", ""))))
                if adds_all and (tt := _total_text(slots, adds_all)):
                    text, nums = tt[0], nums | tt[1]
            if (n := people_count(question)) and (ft := _family_text(slots, n, question)):
                text, nums = text + ft[0], nums | ft[1]
            elif _DISCOUNT_Q.search(last_q) and not re.search(r"这么便宜|挺便宜|真便宜", last_q):
                text, nums = text + FAMILY_SAY, nums | {5.0, 10.0, 15.0, 20.0, 2.0, 3.0, 4.0}
            rest = self._answer_rest(question, known, slots, voice, _PRICE_Q)
            return done(text + gap_tail + rest, "报价", topic_chunks("价格变化"), nums)
        if _WAIT_Q.search(question) and _clauses(question) <= 1 and not u.get("gaps"):
            wait_text = WAIT_ADDON_TEXT if re.search(r"药省保|网上买药|互联网药", question) else WAIT_TEXT
            return done(wait_text, "等待期(模板)", topic_chunks("等待期"))
        if u.get("gaps") and set(u["gaps"]) <= TEMPLATE_ONLY_GAPS and _clauses(question) <= 1:
            lead = "这款产品的保险期间是一年。" if "续保" in u["gaps"] else ""
            return done(lead + gap_tail, "缺口(模板)", topic_chunks("保险期间"))

        for g in u.get("gaps", []):
            must_say.append(GAP_FALLBACK[g])
        # Underwriting questions: only the pre-existing-condition and disclosure clauses, so other
        # items' rules (e.g. an add-on's 20% for nodules) are not dragged into the answer.
        underwriting_only = (not self.direct_answer and intent == "缺口" and "疾病核保" in u.get("gaps", [])
                             and not re.search(r"意外|分别|和|以及|另外|还", question))
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

        facts: list[str] = []
        if "A" in self.features:
            seen_items = list(dict.fromkeys(c.meta["item"] for c in evidence if c.meta.get("item")))
            facts = [f for iid in seen_items for f in item_facts(iid)]
            if facts:  # coverage rules come as citable sentences; the table rows themselves are not shown
                evidence_for_prompt = [c for c in evidence if not c.chunk_id.endswith(":head")]
                evidence = evidence + [Chunk("facts", "保障计划表", "；".join(facts) + "。", "保障计划表（事实句）")]
            else:
                evidence_for_prompt = evidence
        else:
            evidence_for_prompt = evidence
        if "D" in self.features:
            structured |= derived_numbers(question, evidence)
        ctx = grouped_context(evidence_for_prompt)
        calculation = hospital_calculation(question, slots, u.get("items", []))
        if u.get("gaps") or len(u.get("obligations", [])) > 2:
            calculation = None
        if calculation:
            structured |= set(calculation["numbers"])
            trace.append({"stage": "calculate", **calculation})
            trace.append({"stage": "compose", "source": "program", "assumptions": calculation["assumptions"]})
            return done(calculation["text"], "一般医疗测算(模板)", evidence, structured)
        trace.append({"stage": "resolve", "items": list(u.get("items", [])),
                      "evidence": [c.chunk_id for c in evidence], "local_fallbacks": list(u.get("gaps", []))})
        shown = {k: v for k, v in slots.items() if not str(k).startswith("_")}
        history = known.get("_history", "")
        voice_questions = (u.get("_voice_questions") or u.get("obligations")) if voice and not self.direct_answer else None
        voice_questions = voice_questions if isinstance(voice_questions, list) and all(
            isinstance(q, str) and q.strip() for q in voice_questions) else None
        if voice_questions and len(voice_questions) == 1:
            voice_questions = None
        user = (f"已知用户条件：{json.dumps(shown, ensure_ascii=False)}\n"
                + (f"【最近对话】\n{history}\n" if history else "")
                + f"【报价结果】{must_say[0] if intent == '报价' and must_say else '无'}\n"
                f"【必须原样包含的话】{'；'.join(must_say[1:] if intent == '报价' else must_say) or '无'}\n"
                + ("【事实】\n" + "\n".join(f"[F{i + 1}] {f}" for i, f in enumerate(facts)) + "\n" if facts else "")
                + f"【资料】\n{ctx}\n\n用户：{question}")
        user += ("\n【局部兜底】仅对资料不足的待答项说明无法确认，其他项仍须回答；已有病史不能承诺该病赔付。"
                 "每个结论必须绑定对应责任的条件，意外不能套用疾病等待期。额外细节留给追问。")
        if calculation:
            user += "\n【程序计算】" + calculation["text"] + "；结果仅在列明假设成立时适用，不能承诺实际赔款。"
        if voice_questions and not facts:
            user += ("\n【逐项回答槽位】\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(voice_questions, 1))
                     + "\n请输出 JSON：{\"answers\":[{\"id\":1,\"conclusion\":\"...\",\"condition\":\"...\"}]}。"
                     "每个槽位恰好一项；conclusion 直接回答并含被问的具体数字，condition 只放会改变结论的必要条件。"
                     "多项共用的条件只在第一项说一次，后面不要重复；不要照抄问题中的金额或复述用户背景。"
                     "数字与条件不适用时留空；不得填入资料未支持的数字。问总额时只答最终总额。"
                     "不要解释计算过程，不要加未问的细则。")
            user += ("\n【短答约束】每个conclusion只写一个直接结论，不复述问题，不主动报年度额度、医院范围、天数上限。"
                     "condition只写会改变本题结论的条件，共有的免赔额与额度明确说共享一次。"
                     "总回答争取70字，必要事实可超出，不截断；已知既往症免责必须明确说投保前已有病及相关费用不赔。")
        msgs = [{"role": "system", "content": (_DIRECT_SYS if self.direct_answer else _ANSWER_SYS + (VOICE_RULES if voice else "")) + (FACTS_RULE if facts else "")},
                {"role": "user", "content": user}]

        text, violations, attempts = "", [], 0
        # The experimental stream path cannot safely handle JSON fact/slot answers or checks that
        # need the whole reply. It never applies a character cap to released sentences.
        if (on_sentence is not None and not u.get("gaps") and not facts and not voice_questions
                and self.fact_check == "off"):
            spoken, bad = self._stream_generate(msgs, evidence, structured, on_sentence, timings)
            if spoken:
                text = "".join(spoken) + (FOLLOW_UP if voice and bad else "")
                timings["total"] = time.perf_counter() - t0
                timings["streamed_sentences"] = len(spoken)
                return Answer(text, route=route, understanding=u, evidence=evidence, structured=structured,
                              violations=bad, attempts=1, fallback_used=bool(bad), timings=timings)
            if bad:  # the very first sentence failed: nothing was said yet, so rewrite as before
                msgs += [{"role": "user", "content": f"回答不能出现这些问题：{'；'.join(bad)}。请只用资料里的内容回答。"}]
        for attempts in (1, 2):
            t1 = time.perf_counter()
            if facts:
                raw = cloud.chat(msgs, model=self.slow_model, client=self.client, temperature=0.3,
                                 response_format={"type": "json_object"})
                text = assemble_facts(raw, facts) or "这个问题我需要帮您转专员确认一下。"
                timings.setdefault("generate_ttft", time.perf_counter() - t1)
            elif voice_questions:
                r = cloud.chat_stream(msgs, model=self.slow_model, client=self.client, temperature=0.3,
                                      response_format={"type": "json_object"})
                text = assemble_voice_slots(r.text, len(voice_questions))
                timings.setdefault("generate_ttft", r.ttft_s)
            else:
                r = cloud.chat_stream(msgs, model=self.slow_model, client=self.client, temperature=0.3)
                timings.setdefault("generate_ttft", r.ttft_s)
                text = r.text.strip()
            timings[f"generate_{attempts}"] = time.perf_counter() - t1
            # Checked against the routed gaps only: checking against either path's gaps (exp14 first run) turned
            # coverage answers such as "癌症住院能报" into violations whenever one path misread the question.
            violations = (["逐项回答槽位缺失"] if voice_questions and not text else
                          _guard(question, text, evidence, structured, u.get("gaps"), gaps_judged=u.get("_gaps_judged", False),
                                 fact_check=self.fact_check,
                                 features=self.features, client=self.client))
            if voice_questions:
                violations += obligation_guard(question, text, u.get("items", []), u.get("gaps", []))
            if not violations:
                trace.append({"stage": "validate", "attempt": attempts, "violations": [],
                              "slot_count": len(voice_questions or []), "text": text})
                break
            trace.append({"stage": "validate", "attempt": attempts, "violations": list(violations), "text": text})
            # Rewrite as a fresh generation with the problems as constraints: sending the old answer back as a
            # dialogue made the model apologise to the customer ("抱歉，刚才说得不够准确", 第七轮).
            msgs = [msgs[0], {"role": "user", "content": user + "\n\n【注意】回答时避免以下问题："
                              + "；".join(violations) + "。"}]
        fallback = False
        if violations:
            fallback = True
            if not must_say and any("社保抵扣免赔额" in v for v in violations):
                must_say = [GAP_FALLBACK["社保抵扣免赔额"]]
            if "D" in self.features and not must_say and u.get("items"):
                text = spoken_fallback(u["items"], question)
            else:
                text = "".join(must_say) if must_say else (_item_template(u.get("items", [])) or "这个问题我需要帮您转专员确认一下。")
        text = addon_notice(text, evidence)
        trace.append({"stage": "release", "source": "fallback" if fallback else "model", "chars": len(text)})
        timings["total"] = time.perf_counter() - t0
        return Answer(text, route=route, understanding=u, evidence=evidence, structured=structured,
                      violations=violations, attempts=attempts, fallback_used=fallback, timings=timings)


    def _stream_generate(self, msgs, evidence, structured, on_sentence, timings):
        """Stream the slow model; release each sentence once it passes the rule guards.

        Stops at the first sentence that fails a guard (later sentences are not said: dropping one
        sentence mid-answer leaves dangling connectives, 实验 4). Never truncates by length.
        Returns (spoken sentences, violations)."""
        from agent.session import capability_guard
        from agent.speech import SentenceFeeder

        feeder, spoken, bad = SentenceFeeder(soft_flush=None), [], []
        ev_texts = [c.text for c in evidence]
        def release(piece: str) -> bool:
            v = number_guard(piece, ev_texts, structured).violations + compliance_guard(piece).violations
            if capability_guard(piece)[1]:
                v.append(f"能力护栏：{piece}")
            if v:
                bad.extend(v)
                return False
            spoken.append(piece)
            on_sentence(piece)
            return True

        t1 = time.perf_counter()
        stream = self.client.chat.completions.create(model=self.slow_model, messages=msgs, stream=True, temperature=0.3,
                                                     extra_body={"enable_thinking": False})
        go = True
        try:
            for chunk in stream:
                d = chunk.choices[0].delta.content if chunk.choices else ""
                if not d:
                    continue
                timings.setdefault("generate_ttft", time.perf_counter() - t1)
                for piece in feeder.feed(d):
                    if not release(piece):
                        go = False
                        break
                if not go:
                    break
            if go:
                for piece in feeder.flush():
                    if not release(piece):
                        break
        finally:
            stream.close()  # also close on cancellation raised by the sentence callback
        if spoken:
            current = "".join(spoken)
            with_notice = addon_notice(current, evidence)
            if with_notice.startswith(current) and len(with_notice) > len(current):
                release(with_notice[len(current):])
        timings["generate_1"] = time.perf_counter() - t1
        return spoken, bad


# --- streaming variant used by the voice loop --------------------------------------------------------

FOLLOW_UP = "要不要我接着说说？"


VOICE_RULES = ("\n8. 这是电话通话：把回答组织成用户能在约10秒内听懂的一两句，先给结论，再给不可省略的条件或例外。"
               "用户问了几件事就逐一回答，但每件只说最关键的一句，不要为了短而漏答。"
               "问‘都保什么’等概览，只说主要责任类别和一项关键限制，不逐项朗读保额、比例、医院或床位费；"
               "问某个加油包，不顺带介绍其他产品；问计算结果，不朗读推导过程。"
               "不要重复已说过的事实或附加无关的销售提醒，细则留待用户追问。不要使用‘首先、其次、最后’。")


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
