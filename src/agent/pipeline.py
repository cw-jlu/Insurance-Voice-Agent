"""Slow-lane answering pipeline: understand → route → assemble evidence → generate → guard.

Routing follows the solution design (02 方案 3.4):
- prices come from the quote engine, never from the LLM;
- coverage facts come from the coverage table (R01–R16);
- high-frequency key topics map to fixed clauses (no retrieval);
- everything else goes through semantic + keyword retrieval;
- gap questions get a fixed fallback for the part the data cannot answer.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import numpy as np

from agent import cloud
from agent.guardrail import (
    GAPS, IDENTITY_REPLY, compliance_guard, extract_numbers, gap_guard, gap_of, is_identity_question, number_guard,
)
from agent.knowledge import Chunk, clause_chunks, coverage_items
from agent.quote import ADDONS, Insured, quote
from agent.retrieval import Retriever

COVERAGE = {i["id"]: i for i in coverage_items()}

# --- fixed citations for key topics (curated from 投保须知 / 责任免除 / 费率表 text) ------------------
TOPICS: dict[str, list[tuple[str, str]]] = {
    "总保额": [("投保须知", "国内医疗总保额为600万元")],
    "既往症": [("责任免除", "既往症或先天性疾病，不予理赔"), ("责任免除", "因既往症，性病")],
    "等待期": [("投保须知", "必选计划的等待期为30天"), ("责任免除", "等待期内被保险人确诊疾病")],
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
GAP_FALLBACK["条款清单"] = "具体清单以保险条款为准，我可以把条款发给您，或者帮您转专员确认。"

INTENTS = ["报价", "保障", "投保条件", "免责", "缺口", "身份", "闲聊", "其他"]


def _find(doc: str, needle: str) -> list[Chunk]:
    return [c for c in clause_chunks() if c.doc == doc and needle in c.text]


def topic_chunks(topic: str) -> list[Chunk]:
    out: list[Chunk] = []
    for doc, needle in TOPICS.get(topic, []):
        for c in _find(doc, needle):
            if c not in out:
                out.append(c)
    return out


def item_chunks(item_id: str) -> list[Chunk]:
    it = COVERAGE[item_id]
    name = it["责任名称"]
    head = Chunk(
        chunk_id=f"{item_id}:head",
        doc="保障计划表",
        text=(f"{name}（{it['责任&加油包']}）：人群{it['人群']}；等待期{it['等待期']}；保额{it['保额']}；"
              f"赔付比例{'；'.join(it['赔付比例'])}；免赔额{it['免赔额/免赔天数']}；医院范围{'；'.join(it['医院范围'])}"),
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
 "items": 相关保险责任编号列表，从下面目录中选，最多 3 个,
 "topics": 相关关键主题列表，从 %(topics)s 中选,
 "gaps": 用户问的是否属于数据无法回答的内容，从 %(gaps)s 中选，没有则为空列表,
 "rewrite": 把用户口语改写成保险条款用语的检索查询（例如"以前就有的病"→"既往症"）
}
规则：
- 问价格、多少钱、每月多少 → intent=报价。
- 用户说自己（或家人）患有某种病、做过手术、在吃药，问能不能买/能不能报 → intent=缺口，gaps 含"疾病核保"，topics 含"既往症"和"如实告知"。
- 只是问某类疾病或费用本身在不在保障范围（如职业病、先天性疾病、精神疾病、美容、牙科、分娩、体检），没有说自己患病 → intent=免责，topics 含"其他免责"或"既往症"，gaps 为空。
- 问总共保多少、保额多少、能报什么 → topics 含"总保额"。
- 问职业（如高空作业、司机、矿工）能不能买 → intent=投保条件，topics 含"职业"，gaps 为空。
- 问能不能多买几份、重复投保 → intent=投保条件，topics 含"一人一份"。
- 问续保、停售 → gaps 含"续保"；问医保报销部分能否抵扣/算进免赔额 → gaps 含"社保抵扣免赔额"；问疾病清单、药品清单 → gaps 含"条款清单"。
- 问是不是真人/机器人/AI → intent=身份。
- 年龄：用户说"我妈65岁"时 age=65、relation=父母、gender=女。只填用户明确说出的条件，不要猜。
责任目录：
%(catalog)s"""


def _catalog() -> str:
    return "\n".join(f"{i['id']} {i['责任名称']}（{i['责任&加油包']}）" for i in coverage_items())


def understand(question: str, known: dict, client=None, model: str | None = None) -> dict:
    sys = _UNDERSTAND_SYS % {"intents": INTENTS, "topics": TOPIC_NAMES, "gaps": GAP_NAMES, "catalog": _catalog()}
    user = f"已知条件：{json.dumps(known, ensure_ascii=False)}\n用户：{question}"
    raw = cloud.chat([{"role": "system", "content": sys}, {"role": "user", "content": user}],
                     model=model or cloud.FAST_MODEL, client=client, temperature=0,
                     response_format={"type": "json_object"})
    try:
        u = json.loads(raw)
    except json.JSONDecodeError:
        u = {"intent": "其他", "slots": {}, "items": [], "topics": [], "gaps": [], "rewrite": question}
    u.setdefault("slots", {})
    u["items"] = [i for i in u.get("items", []) if i in COVERAGE][:3]
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
7. 如果相关责任属于加油包（资料里写着"××加油包"），要说明这是需要另外选购的加油包，不在基础版里。"""


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


def _guard(question: str, text: str, evidence: list[Chunk], structured: set, gaps: list[str] | None = None) -> list[str]:
    v = number_guard(text, [c.text for c in evidence], structured).violations
    v += gap_guard(question, text, gaps).violations
    v += compliance_guard(text).violations
    return v


def _quote_text(slots: dict) -> tuple[str, set] | None:
    age, social = slots.get("age"), slots.get("has_social")
    if age is None or social is None:
        return None
    y = quote(Insured(int(age), bool(social), gender=slots.get("gender")), "年缴")
    m = quote(Insured(int(age), bool(social), gender=slots.get("gender")), "月缴")
    if not y.ok:
        return "；".join(y.errors), {float(age)}
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


class Pipeline:
    def __init__(self, retriever: HybridRetriever | None, client=None, slow_model: str | None = None,
                 fast_model: str | None = None):
        self.retriever = retriever
        self.client = client or cloud.llm_client()
        self.slow_model = slow_model or cloud.SLOW_MODEL
        self.fast_model = fast_model or cloud.FAST_MODEL

    def answer(self, question: str, known: dict | None = None, voice: bool = False) -> Answer:
        known = dict(known or {})
        t0 = time.perf_counter()
        timings: dict = {}

        if is_identity_question(question):
            return Answer(f"{IDENTITY_REPLY}价格和保障我都能帮您讲清楚，您想先了解哪方面？", route="身份(规则)",
                          timings={"total": time.perf_counter() - t0})

        u = understand(question, known, client=self.client, model=self.fast_model)
        timings["understand"] = time.perf_counter() - t0
        slots = {**known, **{k: v for k, v in (u.get("slots") or {}).items() if v is not None}}
        intent = u.get("intent", "其他")

        if intent == "身份":
            return Answer(f"{IDENTITY_REPLY}价格和保障我都能帮您讲清楚，您想先了解哪方面？", route="身份(模型)",
                          understanding=u, timings={**timings, "total": time.perf_counter() - t0})

        evidence: list[Chunk] = []
        structured: set = {float(v) for v in slots.values() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        must_say: list[str] = []
        route = intent

        if intent == "报价":
            qt = _quote_text(slots)
            if qt is None:
                missing = "年龄" if slots.get("age") is None else "有没有社保"
                ask = "请问被保人今年多大年纪？" if missing == "年龄" else "请问您有社保吗？有没有社保价格不一样。"
                return Answer(ask, route="报价-追问", understanding=u, timings={**timings, "total": time.perf_counter() - t0})
            text, nums = qt
            must_say.append(text)
            structured |= nums
            evidence += topic_chunks("价格变化")

        for g in u.get("gaps", []):
            must_say.append(GAP_FALLBACK[g])
        for item in u.get("items", []):
            evidence += item_chunks(item)
        for t in u.get("topics", []):
            if t == "家庭单优惠":
                evidence.append(Chunk("family", "费率表", FAMILY_RULE, "费率表 家庭单用户优享价格规则说明"))
            else:
                evidence += topic_chunks(t)
        if self.retriever is not None:
            t1 = time.perf_counter()
            evidence += self.retriever.search(u.get("rewrite") or question, k=4)
            timings["retrieve"] = time.perf_counter() - t1
        # de-duplicate, keep order
        seen, ev = set(), []
        for c in evidence:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                ev.append(c)
        evidence = ev

        ctx = "\n".join(f"[{i + 1}] {c.text}（出处：{c.source}）" for i, c in enumerate(evidence))
        user = (f"已知用户条件：{json.dumps(slots, ensure_ascii=False)}\n"
                f"【报价结果】{must_say[0] if intent == '报价' and must_say else '无'}\n"
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
            violations = _guard(question, text, evidence, structured, u.get("gaps"))
            if not violations:
                break
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"上面的回答有问题：{'；'.join(violations)}。请只用资料里的内容重写。"}]
        fallback = False
        if violations:
            fallback = True
            text = "".join(must_say) if must_say else (_item_template(u.get("items", [])) or "这个问题我需要帮您转专员确认一下。")
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
