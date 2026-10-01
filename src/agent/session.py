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
RETAIN = "理解，不过基础版有社保的话一年只要几百元，保障到几百万。您要是愿意，我可以只花半分钟说下重点？"
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
    refusals: int = 0
    risk: dict = field(default_factory=lambda: {"gap_hits": 0, "fallbacks": 0})
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

        if is_identity_question(user_text):
            act, u = "身份", {"slots": {}}
        else:
            known = {**self.profile, "_history": hist}
            with ThreadPoolExecutor(2) as ex:
                fu = ex.submit(understand, user_text, known, self.pipe.client, self.pipe.fast_model)
                fa = ex.submit(small_model.classify_act, user_text, hist, self.pipe.client)
                u, act = fu.result(), fa.result()
            # Identity only when both models agree; one model alone misreads answers like "是给我自己".
            if act == "身份" and u.get("intent") != "身份":
                act = "提供信息" if any(v is not None for v in (u.get("slots") or {}).values()) else "提问"
        timings["understand"] = time.perf_counter() - t0
        events: list[str] = self._update_profile(u.get("slots") or {})
        if any(e.startswith(("纠正:age", "纠正:has_social", "纠正:relation")) for e in events) and self.flags["quoted"]:
            self.flags["quoted"] = False  # price depended on the corrected fact: quote again

        if act == "身份":
            self.flags["ai"] = True
            step, ev = self._advance(already_asked=False)
            text = IDENTITY_REPLY + "价格和保障我都能帮您讲清楚。" + step
            return self._finish(text, act, events + ["承认AI"] + ev, timings, t0)
        if act in ("要人工", "投诉"):
            self.flags["handoff"], self.ended, self.stage = True, True, "结束"
            return self._finish(HANDOFF, act, ["转人工"], timings, t0)
        if act == "拒绝":
            self.refusals += 1
            if self.refusals >= 2 or self.stage in ("投保",):
                self.ended, self.stage = True, "结束"
                return self._finish(GOODBYE, act, ["结束:拒绝"], timings, t0)
            return self._finish(RETAIN, act, ["挽留一次"], timings, t0)

        if u.get("gaps"):
            self.risk["gap_hits"] += 1

        if act == "同意投保":
            if self.flags["quoted"] and self.flags["limits"] and self.flags["disclosure"]:
                self.flags["consent"] = self.flags["link"] = True
                self.ended, self.stage = True, "收尾"
                return self._finish(LINK_SENT, act, ["同意", "发送链接"], timings, t0)
            # gate: not everything required before sending a link has been said yet
            step, ev = self._advance(already_asked=False)
            return self._finish(("好的，" + step) if step else "好的。", act, ev + ["门禁:暂不发链接"], timings, t0)

        answer_text, asked = "", False
        needs_answer = act in ("提问", "异议") or (u.get("intent") in ("报价", "保障", "免责", "缺口", "优惠", "投保条件")
                                                   and act not in ("提供信息", "附和"))
        if needs_answer:
            known = {**self.profile, "_history": hist}
            a = self.pipe.answer(user_text, known, voice=self.voice, agent_steps=self.agent_steps, understanding=u)
            answer_text = a.text
            asked = answer_text.rstrip().endswith(("？", "?"))
            if a.fallback_used:
                self.risk["fallbacks"] += 1
            if a.route == "报价" or ("报价" in (a.route or "") and not a.route.endswith("追问")):
                self.flags["quoted"] = True
            if any(w in answer_text for w in ("免赔", "等待期", "既往症")):
                self.flags["limits"] = self.flags["limits"] or "免赔" in answer_text and "等待期" in answer_text
            timings["answer"] = a.timings.get("total")
            events.append(f"回答:{a.route}")
        if self.risk["gap_hits"] >= 3 and not self.flags["handoff"]:
            self.flags["handoff"], self.ended, self.stage = True, True, "结束"
            return self._finish((answer_text + "这些情况比较细，" + HANDOFF).strip(), act, events + ["转人工:风险累积"], timings, t0)

        from agent.pipeline import FOLLOW_UP

        if answer_text.endswith(FOLLOW_UP):  # the flow's own next question replaces the generic follow-up
            answer_text, asked = answer_text[: -len(FOLLOW_UP)], False
        step, ev = self._advance(already_asked=asked)
        if step and answer_text and step in answer_text:
            step = ""
        prefix = "明白，" if any(e.startswith("纠正:") for e in events) and not answer_text else ""
        text = (answer_text + step) if answer_text else (prefix + step if step else "好的，您还想了解哪方面？")
        return self._finish(text, act, events + ev, timings, t0)

    def _finish(self, text: str, act: str, events: list[str], timings: dict, t0: float) -> TurnResult:
        self._say(text)
        timings["total"] = time.perf_counter() - t0
        return TurnResult(text=text, stage=self.stage, act=act, events=events, timings=timings)
