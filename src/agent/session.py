"""Large chain: one CallSession per outbound call.

The sales flow is a state machine run by code; the per-turn answer (small chain) is
``Pipeline.answer``. Gates between stages, and every compliance sentence, are deterministic:

    开场 → 需求 → 报价 → 讲限制 → 健康告知 → 投保 → 收尾
    (any stage) 提问/异议 → answer, then resume the stage; 要人工/投诉 → 转人工; 两次拒绝 → 结束

Gates:
- 报价 needs age and has_social (asked one at a time otherwise).
- 发投保链接 needs: price quoted, main limits told, 如实告知 told, and the user's explicit consent.
- After the user refuses twice, no more selling: polite close.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from agent import small_model
from agent.guardrail import IDENTITY_REPLY, is_identity_question
from agent.pipeline import Pipeline, _quote_text, understand

OPENING = ("您好，我是众安保险的AI助理，本次通话会录音。看到您之前了解过尊享e生百万医疗，"
           "想花一分钟跟您介绍一下，请问是给您自己了解吗？")
LIMITS = "提醒您一下，一般住院一年有1万元免赔额，首年生病住院有30天等待期，投保前就有的病不赔。"
DISCLOSURE = "投保时需要如实告知健康情况，能不能投保以核保结果为准。您这边有没有正在治疗的疾病？"
ASK_LINK = "需要我把投保链接发到您手机上吗？投保成功后次日零时生效。"
LINK_SENT = "好的，投保链接已经发到您手机上了，您按页面提示填写健康告知就可以。感谢您的时间，再见。"
RETAIN = "理解。我只占用您半分钟说个重点：基础版总保额600万，大病医疗没有免赔额。您要是愿意听，我简单说一下？"
GOODBYE = "好的，不打扰您了，祝您生活愉快，再见。"
HANDOFF = "好的，我马上为您转接人工专员，刚才的情况我会一并告诉专员，您不用重复说。"
ASK = {"relation": "请问是给您自己买，还是给家人买？", "age": "请问被保人今年多大年纪？",
       "has_social": "请问有社保吗？有没有社保价格不一样。"}


@dataclass
class TurnResult:
    text: str
    stage: str
    act: str
    events: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)


@dataclass
class CallSession:
    pipe: Pipeline
    agent_steps: int = 0
    voice: bool = True
    stage: str = "开场"
    profile: dict = field(default_factory=dict)
    profile_log: list = field(default_factory=list)
    history: list = field(default_factory=list)  # (speaker, text); agent lines are what was actually said
    flags: dict = field(default_factory=lambda: {"recording": False, "ai": False, "quoted": False, "limits": False,
                                                 "disclosure": False, "consent": False, "link": False, "handoff": False})
    agent_budget_s: float = 4.0
    refusals: int = 0
    risk: dict = field(default_factory=lambda: {"gap_hits": 0, "fallbacks": 0})
    gap_topics: dict = field(default_factory=dict)
    pending_handoff_offer: bool = False
    ended: bool = False

    # ------------------------------------------------------------------ helpers
    def _history_text(self, n: int = 6) -> str:
        return "\n".join(f"{s}：{t}" for s, t in self.history[-n:])

    def _say(self, text: str) -> None:
        self.history.append(("坐席", text))

    def _update_profile(self, slots: dict) -> list[str]:
        events = []
        for k in ("age", "has_social", "pay", "relation", "gender", "family_size"):
            v = slots.get(k)
            if v is None:
                continue
            if k in self.profile and self.profile[k] != v:
                events.append(f"纠正:{k}:{self.profile[k]}→{v}")
            self.profile[k] = v
            self.profile_log.append((len(self.history), k, v))
        return events

    def _next_missing(self) -> str | None:
        for k in ("relation", "age", "has_social"):
            if self.profile.get(k) is None:
                return k
        return None

    def _advance(self, already_asked: bool) -> tuple[str, list[str]]:
        """The sales step for this turn, decided by the state machine (gates enforced here)."""
        events = []
        if not already_asked and (k := self._next_missing()):
            self.stage = "需求"
            return ASK[k], events
        if self._next_missing():
            return "", events
        if not self.flags["quoted"]:
            qt = _quote_text(self.profile)
            self.stage = "报价"
            if qt is None:
                return "", events
            self.flags["quoted"] = True
            events.append("报价")
            return qt[0], events
        if not self.flags["limits"]:
            self.flags["limits"] = True
            self.stage = "讲限制"
            events.append("提示限制")
            return LIMITS, events
        if not self.flags["disclosure"]:
            self.flags["disclosure"] = True
            self.stage = "健康告知"
            events.append("提示如实告知")
            return DISCLOSURE, events
        self.stage = "投保"
        return ASK_LINK, events

    # ------------------------------------------------------------------ public
    def opening(self) -> str:
        self.flags["recording"] = self.flags["ai"] = True
        self._say(OPENING)
        return OPENING

    def turn(self, user_text: str) -> TurnResult:
        t0 = time.perf_counter()
        self.history.append(("用户", user_text))
        hist = self._history_text()
        timings: dict = {}

        # Understanding always runs, so facts stated in the same sentence are never lost (第五轮: an identity
        # question that skipped extraction made the agent ask the user's age again and again).
        known = {**self.profile, "_history": hist}
        with ThreadPoolExecutor(3) as ex:
            fu = ex.submit(understand, user_text, known, self.pipe.client, self.pipe.fast_model)
            fa = ex.submit(small_model.classify_act, user_text, hist, self.pipe.client)
            fg = ex.submit(small_model.detect_gaps, user_text, self.pipe.client)
            u, act, gaps_now = fu.result(), fa.result(), fg.result()
        # Gaps are judged on this sentence alone: with history, a gap asked earlier "sticks" to every later turn.
        u["gaps"] = gaps_now
        if "疾病核保" in gaps_now and u.get("intent") not in ("缺口",):
            u["intent"] = "缺口"
        rule_identity = is_identity_question(user_text)
        # Identity: the rule, or both models agreeing (one model alone misread "我觉得有医保就够了").
        if rule_identity or (act == "身份" and u.get("intent") == "身份"):
            act = "身份"
        elif act == "身份":
            act = "提问"
        timings["understand"] = time.perf_counter() - t0
        events: list[str] = self._update_profile(u.get("slots") or {})
        if any(e.startswith(("纠正:age", "纠正:has_social", "纠正:relation")) for e in events) and self.flags["quoted"]:
            self.flags["quoted"] = False  # price depended on the corrected fact: quote again

        offered_handoff = self.pending_handoff_offer and self._last_agent_text().endswith(OFFER_HANDOFF)
        if act in ("要人工", "投诉") or (offered_handoff and not self._last_agent_asked_link() and _yes(user_text)):
            self.flags["handoff"], self.ended, self.stage = True, True, "结束"
            return self._finish(HANDOFF, act, events + ["转人工"], timings, t0)
        self.pending_handoff_offer = False
        if act == "拒绝" and re.search(r"考虑|想想|再说|看看|发.{0,4}(介绍|资料|链接)", user_text):
            act = "异议"  # "let me think about it" is hesitation, not a refusal
        if act == "拒绝":
            self.refusals += 1
            if self.refusals >= 2 or self.stage == "投保":
                self.ended, self.stage = True, "结束"
                return self._finish(GOODBYE, act, events + ["结束:拒绝"], timings, t0)
            return self._finish(RETAIN, act, events + ["挽留一次"], timings, t0)

        # Consent: an explicit "同意投保", or a yes to our own link offer.
        consent = act == "同意投保" or (self._last_agent_asked_link() and _yes(user_text))
        if consent:
            if self.flags["quoted"] and self.flags["limits"] and self.flags["disclosure"]:
                self.flags["consent"] = self.flags["link"] = True
                self.ended, self.stage = True, "收尾"
                return self._finish(LINK_SENT, "同意投保", events + ["同意", "发送链接"], timings, t0)
            step, ev = self._advance(already_asked=False)  # gate: something required before the link is missing
            return self._finish(("好的，" + step) if step else "好的。", "同意投保", events + ev + ["门禁:暂不发链接"], timings, t0)

        # Gap topics are counted once per topic; repeated insistence on one topic triggers an OFFER, not a transfer.
        new_gaps = [g for g in u.get("gaps", []) if g not in self.gap_topics]
        for g in u.get("gaps", []):
            self.gap_topics[g] = self.gap_topics.get(g, 0) + 1
        self.risk["gap_hits"] = len(self.gap_topics)

        has_question = (act in ("提问", "异议", "身份") or bool(u.get("gaps"))
                        or (QUESTION_MARKERS.search(user_text) is not None and u.get("intent") not in ("闲聊", "其他")))
        answer_text, asked = "", False
        if act == "身份":
            self.flags["ai"] = True
            answer_text = IDENTITY_REPLY + "价格和保障我都能帮您讲清楚。"
            events.append("承认AI")
            has_question = bool(u.get("gaps")) or u.get("intent") not in ("身份", "闲聊", "其他")
        if has_question and act != "附和" and (act != "身份" or u.get("intent") != "身份"):
            known = {**self.profile, "_history": hist}
            steps = self.agent_steps if self.agent_steps and _is_complex(u, user_text) else 0
            a = self.pipe.answer(user_text, known, voice=self.voice, agent_steps=steps,
                                 agent_budget_s=self.agent_budget_s, understanding=u)
            answer_text += a.text
            asked = a.text.rstrip().endswith(("？", "?"))
            if a.fallback_used:
                self.risk["fallbacks"] += 1
            if a.route == "报价":
                self.flags["quoted"] = True
            if "免赔" in answer_text and "等待期" in answer_text:
                self.flags["limits"] = True
            timings["answer"] = a.timings.get("total")
            events.append(f"回答:{a.route}" + (f"(Agent {a.timings.get('agent_steps', 0)}步)" if steps else ""))

        if any(self.gap_topics[g] >= 2 for g in u.get("gaps", [])) and not new_gaps:
            # the user keeps pressing on something the data cannot answer: offer a specialist, don't force it
            self.pending_handoff_offer = True
            from agent.pipeline import FOLLOW_UP as _FU

            answer_text = answer_text.replace(_FU, "")
            text = _dedupe(answer_text + OFFER_HANDOFF, self._last_agent_text())
            return self._finish(text, act, events + ["询问是否转人工"], timings, t0)

        from agent.pipeline import FOLLOW_UP

        if answer_text.endswith(FOLLOW_UP):  # the flow's own next question replaces the generic follow-up
            answer_text, asked = answer_text[: -len(FOLLOW_UP)], False
        hesitant = act == "异议" and re.search(r"考虑|想想|再说|看看", user_text)
        step, ev = ("", []) if hesitant else self._advance(already_asked=asked)
        if step == ASK_LINK and self._last_agent_asked_link():
            step = ""  # don't repeat the same offer turn after turn
        if hesitant and not self._last_agent_asked_link() and self.flags["quoted"]:
            step = "您可以先把投保链接存着，页面里有完整的产品介绍，方便的时候再看。需要我发到您手机上吗？"
        prefix = "明白，" if any(e.startswith("纠正:") for e in events) and not answer_text else ""
        text = (answer_text + step) if answer_text else (prefix + step if step else "好的，您还想了解哪方面？")
        return self._finish(_dedupe(text, self._last_agent_text()), act, events + ev, timings, t0)

    def _last_agent_text(self) -> str:
        return next((t for s, t in reversed(self.history) if s == "坐席"), "")

    def _last_agent_asked_link(self) -> bool:
        return "投保链接发到您手机" in self._last_agent_text()

    def _finish(self, text: str, act: str, events: list[str], timings: dict, t0: float) -> TurnResult:
        text, removed = capability_guard(text)
        if removed:
            events.append("能力护栏:" + "；".join(removed))
        self._say(text)
        timings["total"] = time.perf_counter() - t0
        return TurnResult(text=text, stage=self.stage, act=act, events=events, timings=timings)


# --- turn helpers ----------------------------------------------------------------------------------

QUESTION_MARKERS = re.compile(r"吗|呢|多少|多久|几|什么|怎么|哪|能不能|是不是|可不可以|有没有|咋|？|\?")
_YES = re.compile(r"^(那)?(好|好的|好吧|行|可以|嗯|对|是的|发吧|发过来|发我|麻烦|那就|没问题|ok|OK)|发(过来|给我|我|吧)|可以发|行啊|好啊")
OFFER_HANDOFF = "这个问题需要专员根据条款确认，需要我帮您转接人工专员吗？"

# Actions the system can actually perform: quote, explain, send the purchase-link SMS, transfer to a human.
_NO_CAPABILITY = re.compile(r"[^。！？]*(微信|邮寄|快递|寄给您|寄资料|发邮件|邮箱|优惠券|额外优惠|给您打折|申请折扣|上门|回访您|"
                            r"(发|寄|传)(给)?(您|你)?.{0,4}(介绍|资料|材料|宣传|条款)|(介绍|资料|材料|宣传).{0,3}(发|寄|传)(给|到)?(您|你))[^。！？]*[。！？]?")


def capability_guard(text: str) -> tuple[str, list[str]]:
    removed = [m.group(0) for m in _NO_CAPABILITY.finditer(text)]
    if not removed:
        return text, []
    cleaned = _NO_CAPABILITY.sub("", text).strip()
    if not cleaned:
        cleaned = "我可以把投保链接发到您手机上，页面里有完整的产品介绍。"
    return cleaned, removed


def _yes(text: str) -> bool:
    return bool(_YES.search(text.strip())) and not re.search(r"不(用|要|需要)|算了|别", text)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[。！？?])", text) if s.strip()]


def _dedupe(text: str, last_agent: str = "") -> str:
    """Drop repeated sentences within this reply, and flow sentences repeated verbatim from the last reply."""
    seen, out = set(), []
    flow = {ASK[k] for k in ASK} | {LIMITS, DISCLOSURE}
    last = set(_sentences(last_agent))
    for s in _sentences(text):
        key = re.sub(r"[，。！？\s]", "", s)
        if key in seen or (s in last and any(s in f for f in flow)):
            continue
        seen.add(key)
        out.append(s)
    return "".join(out)


def _is_complex(u: dict, text: str) -> bool:
    """Route to the evidence agent only for questions that need several pieces of evidence."""
    if "疾病核保" in u.get("gaps", []):
        return False  # underwriting answers use fixed clauses only
    parts = len(re.findall(r"[，,、]|和|还有|以及|另外|呢", text))
    return len(u.get("items", [])) >= 2 or (parts >= 2 and len(QUESTION_MARKERS.findall(text)) >= 2)
