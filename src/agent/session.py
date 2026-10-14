"""Large chain: one CallSession per outbound call.

The sales flow is a state machine run by code; the per-turn answer (small chain) is
``Pipeline.answer``. Gates between stages, and every compliance sentence, are deterministic:

    开场 → 需求 → 报价 → 讲限制 → 健康告知 → 投保 → 收尾
    (any stage) 提问/异议 → finish the tracked consultation before resuming sales;
    要人工/投诉 → 转人工; 两次拒绝 → 结束

Gates:
- 报价 needs age and has_social (asked one at a time otherwise).
- 发投保链接 needs: price quoted, main limits told, 如实告知 told, and the user's explicit consent.
  Consent given before the other gates are met is remembered; the link goes out as soon as they are met
  (第六轮: 5 calls asked again after the user had agreed). Hesitation ("考虑一下""先挂了""商量商量") is never
  consent, whatever the classifier says (第六轮: 2 links sent after "嗯……那我再考虑考虑吧，先挂了").
- After the user refuses twice, no more selling: polite close.
"""

from __future__ import annotations

import re
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from agent import small_model
from agent.fast_lane import FAST_PHRASES, rule_ack, model_ack, waiting_reply, answer_while_waiting
from agent.guardrail import IDENTITY_REPLY, is_identity_question
from agent import consultation as consult
from agent.dialogue import _BACKCHANNEL
from agent.pipeline import Pipeline, _quote_text, combine_gaps, gaps_to_check, understand

OPENING = ("您好，我是众安保险的AI助理，本次通话会录音。看到您之前了解过尊享e生百万医疗，"
           "想花一分钟跟您介绍一下，请问是给您自己了解吗？")
LIMITS = "提醒您一下，一般住院一年有1万元免赔额，首年生病住院有30天等待期，投保前就有的病不赔。"
DISCLOSURE = "投保时需要如实告知健康情况，能不能投保以核保结果为准。您这边有没有正在治疗的疾病？"
ASK_LINK = "需要我把投保链接发到您手机上吗？投保成功后次日零时生效。"
LINK_SENT = "好的，投保链接已经发到您手机上了，您按页面提示填写健康告知就可以。感谢您的时间，再见。"
RETAIN = "理解。我只占用您半分钟说个重点：基础版总保额600万，大病医疗没有免赔额。您要是愿意听，我简单说一下？"
GOODBYE = "好的，不打扰您了，祝您生活愉快，再见。"
HANDOFF = "好的，我马上为您转接人工专员，刚才的情况我会一并告诉专员，您不用重复说。"
POST_HANDOFF = "我已记录转人工请求，并附上刚才的问题和已有信息。"
ASK = {"relation": "请问是给您自己买，还是给家人买？", "age": "请问被保人今年多大年纪？",
       "has_social": "请问有社保吗？有没有社保价格不一样。"}


@dataclass
class TurnResult:
    text: str
    stage: str
    act: str
    events: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    playback_id: int | None = None


@dataclass
class CallSession:
    pipe: Pipeline
    agent_steps: int = 0
    voice: bool = True
    fast_lane: bool = True
    waiting_reply_enabled: bool = True
    waiting_reply_delay_s: float = 1.5
    playback_tracking: bool | None = None  # Voice defaults to receipts; text mode commits text.
    stage: str = "开场"
    profile: dict = field(default_factory=dict)
    profile_log: list = field(default_factory=list)
    history: list = field(default_factory=list)  # (speaker, text); agent lines are what was actually said
    flags: dict = field(default_factory=lambda: {"recording": False, "ai": False, "quoted": False, "limits": False,
                                                 "disclosure": False, "consent": False, "link": False, "handoff": False})
    agent_budget_s: float = 2.5
    refusals: int = 0
    risk: dict = field(default_factory=lambda: {"gap_hits": 0, "fallbacks": 0})
    gap_topics: dict = field(default_factory=dict)
    pending_handoff_offer: bool = False
    handoff_request: dict | None = None
    consultation: dict = field(default_factory=dict)
    _subject_since: int = 0  # User-message index since the latest explicit insured-person replacement.
    ended: bool = False
    _hypothetical: bool = False
    _comparison: bool = False
    _family_mention: bool = False
    _playback_counter: int = 0
    _playback_active: int | None = None
    _playback_records: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.playback_tracking is None:
            self.playback_tracking = self.voice

    # ------------------------------------------------------------------ helpers
    def _history_text(self, n: int | None = None) -> str:
        # Full confirmed history is the default. Explicit n is only for controls.
        history = [h for h in self.history if h[1]]
        if n is not None:
            history = history[-n:] if n > 0 else []
        return "\n".join(f"{s}：{t}" for s, t in history)

    def _say(self, text: str) -> None:
        if self.playback_tracking and self._playback_active is not None:
            record = self._playback_records[self._playback_active]
            record['generated'] = text
            record['generation_done'] = True
        else:
            self.history.append(("坐席", text))

    def _begin_playback(self) -> None:
        if not self.playback_tracking:
            return
        if self._playback_active is not None:
            self.interrupt_playback(self._playback_active)
        self._playback_counter += 1
        self._playback_active = self._playback_counter
        self._playback_records[self._playback_active] = {
            'index': len(self.history), 'generated': '', 'heard': '', 'closed': False,
            'generation_done': False, 'flags_before': dict(self.flags)}
        self.history.append(('坐席', ''))  # Reserve order; empty pending text never enters prompts.

    def confirm_playback(self, playback_id: int, heard: str, complete: bool = False) -> bool:
        record = self._playback_records.get(playback_id)
        if record is None or record['closed']:
            return False  # Late receipts from cancelled turns must not restore stale text.
        if not record['generated'].startswith(heard) or not heard.startswith(record['heard']):
            raise ValueError('Playback receipt must be a monotonic prefix of generated speech')
        if complete and (not record['generation_done'] or heard != record['generated']):
            raise ValueError('Full playback requires the complete finalized response')
        record['heard'] = heard
        self.history[record['index']] = ('坐席', heard)
        if complete:
            record['closed'] = True
            if record.get('consultation_key') == (self.consultation.get('id'),self.consultation.get('revision')):
                self.consultation['status'] = record['consultation_status_after']
        return True

    def interrupt_playback(self, playback_id: int) -> dict:
        record = self._playback_records.get(playback_id)
        if record is None:
            return {'heard': '', 'dropped': ''}
        if not record['closed']:
            record['closed'] = True
            # Conservatively repeat newly generated disclosures if playback was incomplete.
            for flag in ('quoted', 'limits', 'disclosure'):
                if not record['flags_before'].get(flag, False):
                    self.flags[flag] = False
            if record.get('consultation_key') == (self.consultation.get('id'),self.consultation.get('revision')):
                consult.reopen(self.consultation)
        return {'heard': record['heard'], 'dropped': record['generated'][len(record['heard']):]}

    def _update_profile(self, slots: dict) -> list[str]:
        events = []
        for k in ("age", "has_social", "pay", "relation", "gender", "family_size"):
            v = slots.get(k)
            if v is None:
                continue
            if k == "age" and self._hypothetical:
                continue
            if k in ("relation", "gender") and self._family_mention:
                continue  # "我父亲得过肺癌" mentions a relative; it does not change who is insured (第十轮回归)
            if k == "has_social" and self._comparison:
                continue  # "那比有社保贵了一倍多" is not the user's own condition (第十轮回归)
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
            if getattr(self.pipe,'consultation_tracking',False) is True and not consult.quote_scope_known(
                    self.profile,[t for s,t in self.history if s=='用户']):
                self.stage = '需求'
                return '需要报价的话，您是首次投保必选计划个人单，还是其他投保情况？', ['程序报价:范围待确认']
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
        if self.flags["consent"]:
            self.flags["link"], self.ended, self.stage = True, True, "收尾"
            events.append("发送链接(此前已同意)")
            return LINK_SENT, events
        return ASK_LINK, events

    # ------------------------------------------------------------------ public
    def opening(self) -> str:
        self._begin_playback()
        self.flags["recording"] = self.flags["ai"] = True
        self._say(OPENING)
        return OPENING

    def turn(self, user_text: str, on_sentence=None, cancel_check=None) -> TurnResult:
        """``on_sentence``: optional callback receiving each sentence as soon as it can be spoken (answer
        sentences while the model is still generating, then the flow's own sentences). The returned text
        is exactly what was handed to the callback."""
        t0 = time.perf_counter()
        if cancel_check is not None:
            cancel_check()
        self._spoken, self._emit_cb = [], on_sentence
        self.history.append(("用户", user_text))
        self._begin_playback()
        hist = self._history_text()
        timings: dict = {}

        fast_kind, ack = rule_ack(user_text) if self.voice and self.fast_lane else (None, '')
        if ack:
            if cancel_check is not None:
                cancel_check()
            self._emit(ack)
            timings['fast_lane'] = {'source': 'rule', 'kind': fast_kind,
                                    'text': ack, 'emit_s': time.perf_counter() - t0}
        if fast_kind == 'handoff':
            self.handoff_request = {'status': 'requested', 'origin': 'explicit_user',
                                    'question': user_text, 'known': dict(self.profile)}
            if self.consultation:
                self.handoff_request['consultation'] = deepcopy(self.consultation)
            self.flags['handoff'], self.ended, self.stage = True, True, '结束'
            return self._finish(ack, '要人工', ['转人工请求:快车道'], timings, t0)

        # Understanding always runs, so facts stated in the same sentence are never lost (第五轮: an identity
        # question that skipped extraction made the agent ask the user's age again and again).
        tracking = getattr(self.pipe,'consultation_tracking',False) is True
        subject_change = tracking and bool(re.search(r'(?:换成|改成|改为|换给)(?:给)?(?:母亲|父亲|妈妈|爸爸|孩子|女儿|儿子|妻子|丈夫|爱人|配偶|自己|我)',user_text))
        if subject_change:
            self._subject_since = sum(s=='用户' for s,t in self.history)-1
            for name in ('age','has_social','gender','relation'):
                self.profile.pop(name,None)
            self.flags['quoted'] = False
            self.flags['consent'] = False  # Consent for the previous person cannot authorize the replacement.
            if self.consultation.get('status')=='complete':
                self.consultation = {}  # The finished prior-person consultation stays in the trace/history.
            elif self.consultation:
                consult.reopen(self.consultation,('age','has_social','gender','insured_with_social','remaining_deductible'))
        known = {**self.profile, "_history": hist}
        self._hypothetical = bool(HYPOTHETICAL_AGE.search(user_text))
        self._comparison = bool(re.search(r"比(有|没有|没)?(社保|医保)|如果|要是|假如|万一", user_text))
        self._family_mention = (bool(re.search(r"(爸|妈|父亲|母亲|老人|家里人|亲戚).{0,6}(得过|得了|患|有.{0,4}病|病|癌|走了|去世)", user_text))
                                and not re.search(r"给.{0,3}(爸|妈|父亲|母亲)|替.{0,3}(爸|妈)|帮.{0,3}(爸|妈)|(爸|妈|父亲|母亲).{0,6}(买|投保|多少钱)", user_text))
        with ThreadPoolExecutor(4) as ex:  # all independent of each other: the slowest one sets the time
            fu = ex.submit(self.pipe.understand, user_text, known)
            fa = ex.submit(small_model.classify_act, user_text, hist, self.pipe.client)
            fg = (None if getattr(self.pipe, 'post_answer_routing', False)
                  else ex.submit(small_model.detect_gaps, user_text, self.pipe.client))
            fr = ex.submit(self.pipe.retrieve, user_text)
            act = fa.result()
            if self.voice and self.fast_lane and not ack:
                ack = model_ack(act)
                if ack:
                    if cancel_check is not None:
                        cancel_check()
                    self._emit(ack)
                    timings['fast_lane'] = {'source': 'small_model_act', 'kind': 'query',
                                            'text': ack, 'emit_s': time.perf_counter() - t0}
            u, gaps_now = fu.result(), fg.result() if fg is not None else []
            u["_retrieved"] = fr.result()
        if cancel_check is not None:
            cancel_check()  # cancelled partials must not proceed to answer generation
        # Gaps are judged on this sentence alone (with history, a gap asked earlier "sticks" to every later turn),
        # and only when the detector and the understanding module agree: the detector alone misfired on ~9 of
        # 48 calls in 第六轮 (exp12: 46/221 false positives alone, 7/221 combined).
        u["_gaps_check"] = gaps_to_check(u.get("gaps", []), gaps_now)
        u["gaps"], u["_gaps_judged"] = combine_gaps(u.get("gaps", []), gaps_now, user_text), True
        if getattr(self.pipe, 'post_answer_routing', False):
            # The evidence model decides after reading, so an earlier detector
            # cannot replace its answer or trigger the old repeated-gap offer.
            u['gaps'], u['_gaps_check'] = [], []
        if "疾病核保" in u["gaps"] and u.get("intent") not in ("缺口",):
            u["intent"] = "缺口"
        elif u.get("intent") == "缺口" and not u["gaps"]:
            u["intent"] = "保障" if u.get("items") else "其他"
        rule_identity = is_identity_question(user_text)
        # Identity: the rule, or both models agreeing (one model alone misread "我觉得有医保就够了").
        if rule_identity or (act == "身份" and u.get("intent") == "身份"):
            act = "身份"
        elif act == "身份":
            act = "提问"
        if act != "身份" and u.get("intent") == "身份":
            u["intent"] = "其他"  # one model alone misreads e.g. "没有，身体挺好的" (第八轮验收)
        if fast_kind in ('query', 'correction') and act not in ('身份', '拒绝', '要人工', '投诉'):
            act = '提问'  # Explicit consultation must not be lost to an act-classifier miss.
        if subject_change and act not in ('要人工','投诉','身份'):
            act='提问'  # Replacing one insured person is a new consultation, even if the prior person "不买".
        timings["understand"] = time.perf_counter() - t0
        events: list[str] = self._update_profile(u.get("slots") or {})
        # what _update_profile ignored must not reach the answer either (第十轮回归: "要是她70多岁了" was quoted at 70)
        u["slots"] = {k: v for k, v in (u.get("slots") or {}).items()
                      if not ((k == "age" and self._hypothetical) or (k == "has_social" and self._comparison)
                              or (k in ("relation", "gender") and self._family_mention))}
        if any(e.startswith(("纠正:age", "纠正:has_social", "纠正:relation")) for e in events) and self.flags["quoted"]:
            self.flags["quoted"] = False  # price depended on the corrected fact: quote again
        tracking = getattr(self.pipe,'consultation_tracking',False) is True
        changed = [e.split(':')[1] for e in events if e.startswith('纠正:')]
        if tracking and self.consultation and (changed or re.search(r'更正|说错|重新报|重算|改成|换成',user_text)):
            # Reopen BEFORE consent/link gates, including a correction and consent
            # stated in the same sentence.
            consult.reopen(self.consultation,changed)
        explicit_query=tracking and (bool(u.get('obligations')) or (bool(QUESTION_MARKERS.search(user_text)) and bool(re.search(
            r'投保|保费|费率|保障|责任|加油包|医保|社保|医院|赔|能报|津贴|补贴|免赔|等待|保险|多少钱',user_text))))
        if explicit_query and act not in ('身份','要人工','投诉','拒绝'):
            act='提问'

        offered_handoff = self.pending_handoff_offer and self._last_agent_text().endswith(OFFER_HANDOFF)
        accepts_handoff = offered_handoff and not self._last_agent_asked_link() and (_yes(user_text) or ACCEPT_HANDOFF.search(user_text))
        if DO_NOT_CALL.search(user_text):
            self.ended, self.stage = True, "结束"
            return self._finish(GOODBYE, "拒绝", events + ["结束:要求不再来电"], timings, t0)
        if (re.search(r"要投诉|我要举报|骚扰|(?:天天|每天|一直|反复).{0,5}打电话.{0,8}(?:烦|讨厌|受不了)", user_text)
                and not re.search(r"不(?:是|想|要).{0,3}(?:投诉|举报)|没有骚扰", user_text)):
            act = "投诉"
            events.append("投诉优先路由")
        if (re.search(r"转人工|人工客服|找人工|要人工|真人客服|转接人工|转专员|(帮我|给我|那你)转(一下|过去|吧)?", user_text)
                and not re.search(r"不用转|先别转|不要转|别转", user_text)):
            act = "要人工"  # the classifier missed "能帮我转人工客服吗" once (第十轮回归)
        if act in ("要人工", "投诉") or accepts_handoff:
            self.flags["handoff"], self.ended, self.stage = True, True, "结束"
            if getattr(self.pipe,'consultation_tracking',False) is True and self.consultation:
                self.handoff_request = {'status':'requested','origin':'explicit_user',
                    'question':self.consultation['initial_question'],'latest_user':user_text,
                    'known':dict(self.profile),'consultation':deepcopy(self.consultation)}
                self.consultation['status'] = 'handed_off'
                return self._finish(POST_HANDOFF,act,events+['转人工请求:保留原咨询'],timings,t0)
            return self._finish(HANDOFF, act, events + ["转人工"], timings, t0)
        self.pending_handoff_offer = False
        if not subject_change and act in ("异议", "提问", "附和") and re.search(r"不需要|不要了|不买了?|没兴趣|用不着", user_text)                 and not QUESTION_MARKERS.search(user_text):
            act = "拒绝"  # "517啊，太贵了，不需要" is a refusal, not only a price objection (第十轮回归)
        if act == "拒绝" and re.search(r"考虑|想想|再说|看看|发.{0,4}(介绍|资料|链接)", user_text):
            act = "异议"  # "let me think about it" is hesitation, not a refusal
        if DO_NOT_CALL.search(user_text):
            self.ended, self.stage = True, "结束"
            return self._finish(GOODBYE, "拒绝", events + ["结束:要求不再来电"], timings, t0)
        if act == "拒绝":
            self.refusals += 1
            if self.refusals >= 2 or self.stage == "投保":
                self.ended, self.stage = True, "结束"
                return self._finish(GOODBYE, act, events + ["结束:拒绝"], timings, t0)
            return self._finish(RETAIN, act, events + ["挽留一次"], timings, t0)

        if HESITATE.search(user_text) and self.flags["consent"]:
            self.flags["consent"] = False  # an earlier yes is withdrawn by later hesitation
            events.append("撤回同意")
        if OTHER_CHANNEL.search(user_text) and not re.search(r"手机|短信", user_text):
            return self._finish(OTHER_CHANNEL_REPLY, act, events + ["其他渠道:改为询问链接"], timings, t0)
        if _consents(user_text, act, self._last_agent_asked_link()) and not consult.pending(self.consultation) and not explicit_query:
            self.flags["consent"] = True
            if self.flags["quoted"] and self.flags["limits"] and self.flags["disclosure"]:
                self.flags["link"] = True
                self.ended, self.stage = True, "收尾"
                return self._finish(LINK_SENT, "同意投保", events + ["同意", "发送链接"], timings, t0)
            # Gate: something required before the link is missing. Say all of it now rather than one item per turn;
            # the link goes out on the next turn (or right away, if _advance reaches the end).
            steps, ev = [], []
            while True:
                step, e = self._advance(already_asked=False)
                ev += e
                if step:
                    steps.append(step)
                if not step or step in ASK.values() or step == DISCLOSURE or self.ended or '程序报价:范围待确认' in e:
                    break
            return self._finish("好的，" + "".join(steps) if steps else "好的。", "同意投保",
                                events + ev + ["同意(记住)", "门禁:暂不发链接"], timings, t0)
        if act == "同意投保":
            act = "提问" if QUESTION_MARKERS.search(user_text) else "附和"  # classifier said yes, wording did not

        # Gap topics are counted once per topic; repeated insistence on one topic triggers an OFFER, not a transfer.
        new_gaps = [g for g in u.get("gaps", []) if g not in self.gap_topics]
        for g in u.get("gaps", []):
            self.gap_topics[g] = self.gap_topics.get(g, 0) + 1
        self.risk["gap_hits"] = len(self.gap_topics)

        has_question = (act in ("提问", "异议", "身份") or bool(u.get("gaps"))
                        or (QUESTION_MARKERS.search(user_text) is not None and u.get("intent") not in ("闲聊", "其他")))
        if HESITATE.search(user_text) and not QUESTION_MARKERS.search(user_text):
            has_question = False  # "那我再考虑考虑吧，先挂了": nothing to answer, only the keep-the-link offer below
        if tracking and consult.pending(self.consultation) and (act in ('提供信息','提问','异议','同意投保') or changed
                or (act=='附和' and not _BACKCHANNEL.match(user_text.strip()))):
            if act=='附和':
                act='提供信息'
            has_question = True
            events.append('继续原咨询')
        answer_text, asked = "", False
        if act == "身份":
            self.flags["ai"] = True
            answer_text = IDENTITY_REPLY + "价格和保障我都能帮您讲清楚。"
            events.append("承认AI")
            self._emit(answer_text)
            has_question = bool(u.get("gaps")) or u.get("intent") not in ("身份", "闲聊", "其他")
        if has_question and act != "附和" and (act != "身份" or u.get("intent") != "身份"):
            if tracking and not consult.pending(self.consultation):
                self.consultation = consult.start(user_text,f'c{len(self.history)}')
            if tracking:
                consult.add_questions(self.consultation,u.get('obligations',[]))
            known = {**self.profile, "_history": hist}
            if tracking:
                known['_consultation'] = consult.model_context(self.consultation)
                known['_user_texts'] = [t for s,t in self.history if s=='用户']
                known['_condition_user_texts'] = known['_user_texts'][self._subject_since:]
            steps = self.agent_steps if self.agent_steps and _is_complex(u, user_text) else 0
            if tracking and self.voice and self.fast_lane and self.waiting_reply_enabled:
                wait_context=consult.model_context(self.consultation)
                a,bridge = answer_while_waiting(
                    lambda:self.pipe.answer(user_text,known,voice=self.voice,agent_steps=steps,
                                            agent_budget_s=self.agent_budget_s,understanding=u,on_sentence=None),
                    lambda:waiting_reply(user_text,hist,wait_context,self.pipe.client),
                    self._emit,cancel_check,delay_s=self.waiting_reply_delay_s)
                timings['waiting_reply']=bridge
                if bridge['emitted']:
                    answer_text+=bridge['text']
                    events.append('等待承接:小模型')
            else:
                a = self.pipe.answer(user_text, known, voice=self.voice, agent_steps=steps,
                                     agent_budget_s=self.agent_budget_s, understanding=u,
                                     on_sentence=self._emit if self._emit_cb is not None else None)
            if cancel_check is not None:cancel_check()
            if tracking:
                progress = a.timings.get('consultation')
                if progress:
                    self.consultation = deepcopy(progress)
                elif a.outcome.get('action')=='error':
                    self.consultation['status'] = 'error'
                if a.outcome.get('action') in ('answer','clarify') and any(t['kind']=='price' and t['status']=='answered'
                        and type(t.get('quoted_amount')) in (int,float) for t in self.consultation.get('tasks',[])):
                    self.flags['quoted'] = True
                if self.playback_tracking and self._playback_active is not None:
                    record = self._playback_records[self._playback_active]
                    record['consultation_key'] = (self.consultation.get('id'),self.consultation.get('revision'))
                    record['consultation_status_after'] = self.consultation['status']
                    self.consultation['status'] = 'awaiting_playback'
                self._emit(a.text)
                timings['consultation'] = deepcopy(self.consultation)
            # a blocked answer with nothing program-made to say instead: say nothing and let the flow continue,
            # rather than an unrelated "转专员确认" line (第十轮回归)
            answer_text += "" if (a.fallback_used and a.text.startswith("这个问题我需要帮您转专员确认")) else a.text
            asked = a.text.rstrip().endswith(("？", "?"))
            if a.fallback_used:
                self.risk["fallbacks"] += 1
            if a.route == "报价":
                self.flags["quoted"] = True
            if ("免赔" in answer_text and "等待期" in answer_text
                    and ("既往" in answer_text or "投保前" in answer_text)):
                self.flags["limits"] = True
            timings["answer"] = a.timings.get("total")
            timings["answer_detail"] = a.timings
            timings["answer_validation"] = {"route": a.route, "fallback_used": a.fallback_used,
                                             "attempts": a.attempts, "violations": a.violations}
            events.append(f"回答:{a.route}" + (f"(Agent {a.timings.get('agent_steps', 0)}步)" if steps else ""))

            if a.handoff_request is not None:
                # This records a request for the existing transfer integration;
                # it does not pretend the telephony provider has connected anyone.
                self.handoff_request = a.handoff_request
                self.flags['handoff'], self.ended, self.stage = True, True, '结束'
                timings['handoff_request'] = a.handoff_request
                timings['handoff_notice'] = POST_HANDOFF
                return self._finish(answer_text + POST_HANDOFF, act, events + ['后置转人工请求'], timings, t0)
            if a.outcome.get('action') == 'clarify':
                return self._finish(answer_text, act, events + ['追问缺少条件'], timings, t0)
            if a.outcome.get('action') == 'error':
                return self._finish(answer_text, act, events + ['响应异常:暂停销售流程'], timings, t0)
            if tracking:
                # Deliver the finished consultation first; sales may advance on
                # the next user turn, rather than appending another generic quote.
                return self._finish(answer_text,act,events+['原咨询完成:本轮不追加销售'],timings,t0)

        if any(self.gap_topics[g] >= 2 for g in u.get("gaps", [])) and not new_gaps:
            # the user keeps pressing on something the data cannot answer: offer a specialist, don't force it
            self.pending_handoff_offer = True
            from agent.pipeline import FOLLOW_UP as _FU

            answer_text = answer_text.replace(_FU, "")
            text = _dedupe(answer_text + OFFER_HANDOFF, self._last_agent_text())
            return self._finish(text, act, events + ["询问是否转人工"], timings, t0)

        from agent.pipeline import FOLLOW_UP

        if tracking and consult.pending(self.consultation):
            return self._finish(answer_text or '请补充刚才需要确认的信息，我再继续回答原来的问题。',act,
                                events+['咨询未完成:暂停销售'],timings,t0)

        if answer_text.endswith(FOLLOW_UP):  # the flow's own next question replaces the generic follow-up
            answer_text, asked = answer_text[: -len(FOLLOW_UP)], False
        hesitant = act in ("异议", "附和", "提问") and HESITATE.search(user_text)
        # Answer the user's question first. The generic limits reminder can wait until a later
        # turn; explicit consent still goes through the pre-link gate above, which fills it in.
        defer_limits = (bool(answer_text) and self.flags["quoted"] and not self.flags["limits"]
                        and not self._next_missing())
        step, ev = ("", []) if hesitant or defer_limits else self._advance(already_asked=asked)
        if step == ASK_LINK and self._last_agent_asked_link():
            step = ""  # don't repeat the same offer turn after turn
        if hesitant and not self._last_agent_asked_link() and self.flags["quoted"]:
            step = "您可以先把投保链接存着，页面里有完整的产品介绍，方便的时候再看。需要我发到您手机上吗？"
        prefix = "明白，" if any(e.startswith("纠正:") for e in events) and not answer_text else ""
        text = (answer_text + step) if answer_text else (prefix + step if step else "好的，您还想了解哪方面？")
        return self._finish(_dedupe(text, self._last_agent_text()), act, events + ev, timings, t0)

    def _last_agent_text(self) -> str:
        return next((t for s, t in reversed(self.history) if s == "坐席" and t), "")

    def _last_agent_asked_link(self) -> bool:
        return "投保链接发到您手机" in self._last_agent_text()

    def _emit(self, sentence: str) -> None:
        if self._emit_cb is not None:
            self._spoken.append(sentence)
            if self.playback_tracking and self._playback_active is not None:
                self._playback_records[self._playback_active]['generated'] += sentence
            self._emit_cb(sentence)

    def _finish(self, text: str, act: str, events: list[str], timings: dict, t0: float) -> TurnResult:
        # Keep the acknowledgment in returned text/history even without streaming.
        ack = timings.get('fast_lane', {}).get('text', '')
        business_text = text[len(ack):] if ack and text.startswith(ack) else text
        text, removed = capability_guard(business_text)
        if ack:
            text = ack + text
        if removed:
            events.append("能力护栏:" + "；".join(removed))
        if self._emit_cb is not None:
            # Sentences already streamed stay as said; release only what is left (flow steps, templates).
            said = re.sub(r"[，。！？；\s]", "", "".join(self._spoken))
            for s in _sentences(text):
                if re.sub(r"[，。！？；\s]", "", s) not in said:
                    self._emit(s)
            text = "".join(self._spoken)
            self._emit_cb = None
        self._say(text)
        timings["total"] = time.perf_counter() - t0
        return TurnResult(text=text, stage=self.stage, act=act, events=events, timings=timings,
                          playback_id=self._playback_active if self.playback_tracking else None)


def fixed_phrases() -> list[str]:
    """Sentences the agent says verbatim; their audio can be synthesized once, before the call."""
    from agent.pipeline import FOLLOW_UP, GAP_FALLBACK, WAIT_TEXT

    whole = [*FAST_PHRASES, OPENING, LIMITS, DISCLOSURE, ASK_LINK, LINK_SENT, RETAIN, GOODBYE, HANDOFF, OFFER_HANDOFF, FOLLOW_UP, CLAIMS_PROCESS,
             OTHER_CHANNEL_REPLY,
             IDENTITY_REPLY + "价格和保障我都能帮您讲清楚。", WAIT_TEXT, "这款产品的保险期间是一年。", "好的，您还想了解哪方面？",
             *ASK.values(), *GAP_FALLBACK.values()]
    out = []
    for w in whole:
        out += [s for s in re.split(r"(?<=[。！？?])", w) if s.strip()]
    return out


# --- turn helpers ----------------------------------------------------------------------------------

QUESTION_MARKERS = re.compile(r"吗|呢|多少|多久|几|什么|怎么|哪|能不能|是不是|可不可以|有没有|咋|？|\?")
_YES = re.compile(r"^(那)?(好|好的|好吧|行|可以|嗯|对|是的|发吧|发过来|发我|麻烦|那就|没问题|ok|OK)|发(过来|给我|我|吧)|可以发|行啊|好啊")
HYPOTHETICAL_AGE = re.compile(r"等我|到时候|以后|将来|老了|万一|要是|如果|假如|\d+多|[一二三四五六七八九十]+多(岁|了)")
DO_NOT_CALL = re.compile(r"别(再)?打(电话|了|来|过来|扰)|不要(再)?打(电话|了|来)|以后别(再)?(打|来电)|别再来电|不要再来电|拉黑|别骚扰")
OTHER_CHANNEL = re.compile(r"微信|邮箱|邮件|QQ|qq")
OTHER_CHANNEL_REPLY = "微信和邮箱这边发不了。我可以把投保链接用短信发到您手机上，页面里有完整的产品介绍，需要吗？"
# Hesitation or postponement: never consent (第六轮 #7 "回去跟家里人商量商量，到时候再联系你们吧" got a link).
HESITATE = re.compile(r"考虑|想想|想一想|再说|先挂|商量|回头|到时候|再联系|联系你们|改天|以后再|过两天|过几天|不急|比较|"
                      r"再看看|看看再|还没想好|没想好|等我|问问家|问一下家|跟家里|和家里")
_NEGATE = re.compile(r"不(用|要|需要)(发|买|了吧)|不买|不想买|先别|别发|算了|没兴趣|不需要")
_SEND_REQUEST = re.compile(r"发(过来|给我|我|一个|个)?(链接|过来)|链接发|把链接|发我手机|发到我手机|发短信|"
                           r"(你|那你|那就)?发(给)?我吧|你发吧|发来(吧|看看)")
ACCEPT_HANDOFF = re.compile(r"(那|行|好|可以)?(你)?(帮我)?转(吧|人工|专员|过去)|转一下")
# Explicit agreement to receive the link or to buy, in the user's own words.
_EXPLICIT_YES = re.compile(r"发(过来|给我|我|吧|一个|个|到我|链接)|可以发|那就买|我买|买一份|就它了|就这个|要买|"
                           r"怎么(买|投保|投)|(那就|就|我要|那我|给(她|他|我|孩子))投保吧|给我办|办一个|上车")
_SHORT_YES = re.compile(r"^(嗯|那|那就)?[，,]?(好|好的|好吧|好啊|行|行吧|行啊|可以|可以的|可以啊|没问题|成|中|ok|OK|要|需要)[。！!，,～~]*$")
OFFER_HANDOFF = "这个问题需要专员根据条款确认，需要我帮您转接人工专员吗？"

# Actions the system can actually perform: quote, explain, send the purchase-link SMS, transfer to a human.
# Claims-process details (APP, online application, payout timing) are not in the data: 第六轮 "下载APP在线申请，审核通过就打款".
_NO_CAPABILITY = re.compile(r"[^。！？]*(微信|邮寄|快递|寄给您|寄资料|发邮件|邮箱|优惠券|额外优惠|给您打折|申请折扣|上门|回访您|"
                            r"回电|回您|给您回|(再|稍后|晚点|回头|之后)(给您|跟您)?(打|联系|回复)|联系您|通知您|告诉您结果|"
                            r"查到.{0,6}(告诉|联系|回|通知)|帮您查(一下|查|询)|"
                            r"下载|APP|App|app|小程序|公众号|在线申请|线上申请|打款|到账|秒赔|审核通过就|"
                            r"理赔(很|特别|非常|挺)?(快|简单|方便|容易|不麻烦)|"
                            r"(发|寄|传)(给)?(您|你)?.{0,4}(介绍|资料|材料|宣传|条款|清单)|"
                            r"(介绍|资料|材料|宣传|条款|清单).{0,3}(发|寄|传)(给|到)?(您|你))[^。！？]*[。！？]?")


CLAIMS_PROCESS = "理赔的具体流程和材料要以条款为准，需要的话我帮您转专员详细说明。"
_PROCESS = re.compile(r"下载|APP|App|app|小程序|公众号|在线申请|线上申请|打款|到账|秒赔|审核通过就|理赔")


def capability_guard(text: str) -> tuple[str, list[str]]:
    # saying what we cannot do ("微信和邮箱这边发不了") is not a promise
    keep = lambda m: re.search(r"发不了|没法|无法|不能|不支持|做不到|没有(额外|其他)?(优惠|活动)", m.group(0))
    removed = [m.group(0) for m in _NO_CAPABILITY.finditer(text) if not keep(m)]
    if not removed:
        return text, []
    cleaned = _NO_CAPABILITY.sub(lambda m: m.group(0) if keep(m) else "", text).strip()
    if any(re.search(r"投保|购买|买", r) and not re.search(r"理赔|赔款|打款|到账", r) for r in removed):
        cleaned = "投保的话，我可以把投保链接用短信发到您手机上。" + cleaned  # purchase channel, not claims
    elif any(_PROCESS.search(r) for r in removed):
        cleaned = CLAIMS_PROCESS + cleaned  # the question still gets an answer
    elif not cleaned:
        cleaned = "我可以把投保链接发到您手机上，页面里有完整的产品介绍。"
    return cleaned, removed


def _yes(text: str) -> bool:
    return bool(_YES.search(text.strip())) and not re.search(r"不(用|要|需要)|算了|别", text) and not HESITATE.search(text)


def _consents(text: str, act: str, asked_link: bool) -> bool:
    """Consent to the purchase link. Only explicit wording counts; a leading "嗯" or the classifier alone
    does not. A bare "好的/行/可以" counts only as the answer to our own link offer."""
    t = text.strip()
    if _NEGATE.search(t) or OTHER_CHANNEL.search(t):
        return False
    if re.search(r"(吗|吧)[？?。]?$", t) and re.search(r"是不是|是直接|如果|要是|怎么|是要|能(正常)?(投保|买)|可以(投保|买)", t):
        return False  # "如果我要买的话，是直接给我发个链接吗？" asks how buying works; "能发个链接给我看看吗" is a request
    if _SEND_REQUEST.search(t):
        return True  # the user asks for the link itself; sending it is what they asked for, even "看看再说"
    if HESITATE.search(t):
        return False
    if _EXPLICIT_YES.search(t):
        return True
    return asked_link and (bool(_SHORT_YES.match(t)) or act == "同意投保")


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
    if "疾病核保" in u.get("gaps", []) or u.get("intent") in ("报价", "优惠", "身份"):
        return False  # underwriting answers use fixed clauses; prices come from the quote engine
    # count independent questions: question words / "吗" / "呢" separated by clause breaks
    clauses = [c for c in re.split(r"[，,。；;？?]", text) if QUESTION_MARKERS.search(c)]
    return len(u.get("items", [])) >= 2 or len(clauses) >= 2
