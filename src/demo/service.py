"""Isolated sessions, cooperative cancellation and browser playback receipts."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import threading
import time
import uuid
import wave
from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path

from agent import cloud
from agent.evidence_answer import EvidenceAnswerer, grouped_evidence, source_registry
from agent.evidence_pipeline import EvidencePipeline
from agent.session import CallSession, TurnResult, HANDOFF, LINK_SENT, _sentences
from agent import consultation as consult
from agent.speech import extract_slots
from demo.telemetry import Timeline, ObservedPipeline
from demo.speech import OUTBOUND_OPENING, PRODUCT_INTRO, PRODUCT_DETAIL, SpeechCache, SpeechNotPreparedError, WebRealtimeTTS
from demo.outbound import ACK, CLARIFY, select_action, script_remainder

ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / 'experiments/results/exp44_voice_current/voice'
CORE_TAG = 'v0.1.0-core-20261014'
DEMO_TAG = 'v0.3.0-duplex-20261014'
SCENARIOS = [
    {'id':'free','name':'现场自由提问','description':'评委提出新问题，实时识别、回答并查看全过程。','mode':'live'},
    {'id':'replay','name':'完整对话回放','description':'7轮已保存的真实实验，离线查看报价、更正与人工请求。','mode':'replay'},
    {'id':'price','name':'报价与条件更正','description':'先报价，再更正医保身份，检查差额是否更新。','mode':'live',
     'questions':['我35岁有医保，首次买必选计划个人单，年缴、月缴和全年差额各是多少？','说错了，我没有医保，三个金额重新算。']},
    {'id':'coverage','name':'一次问多件事','description':'住院、靶向药与质子治疗，分别问清适用条件。','mode':'live',
     'questions':['如果以后得了癌症，住院、靶向药和质子重离子各怎么赔？各有没有免赔额，医院有要求吗？']},
    {'id':'claim','name':'补条件再测算','description':'实际赔付缺必要条件时，继续原问题而非跳到销售。','mode':'live',
     'questions':['住院合规费用15000元，没走医保，这次实际能赔多少？','按有医保身份投保，今年剩余免赔4000元。']},
    {'id':'history','name':'病史与安全边界','description':'区分已有病史、无关意外和需要人工核实的事项。','mode':'live',
     'questions':['如果投保前有高血压，之后发生无关的交通意外住院，是不是一定都不赔？请说明条件。','转人工客服。']},
]


def capabilities():
    qwen = all(os.getenv(k) for k in ('DASHSCOPE_API_KEY','DASHSCOPE_BASE_URL'))
    return {'replay':True,'qwen':bool(qwen),'deepseek':bool(qwen and os.getenv('DEEPSEEK_API_KEY')),
            'asr':all(bool(os.getenv(k)) for k in ('TENCENT_APP_ID','TENCENT_SECRET_ID','TENCENT_SECRET_KEY')),
            'cloud_tts':bool(qwen and os.getenv('DASHSCOPE_WS_URL'))}


class WebAnswerer(EvidenceAnswerer):
    def __init__(self,client,model):
        super().__init__(client.with_options(timeout=12,max_retries=0))
        self.model=model
        self._context=threading.local()

    def answer(self,question,known,evidence,**kwargs):
        self._context.question=question
        self._context.consultation=deepcopy(known.get('_consultation',{}))
        return super().answer(question,known,evidence,**kwargs)

    def call(self,system,user,model):
        if self.model==cloud.QWEN_ANSWER_MODEL:
            # Qwen's comparison mode otherwise follows the earlier single-turn
            # example and omits v4 consultation metadata. No answers are repaired.
            system+='''
【最终输出结构：持续咨询v4】
不要采用前面的单轮示例结构。最终JSON必须同时包含segments、unknowns、outcome、consultation。
consultation必须包含非空tasks及facts数组。输入_consultation.tasks的每个id全部保留，同一问题使用原id。
每个segments条目包含text、source_ids、derivation、task_ids；task_ids必须列实际回答或追问的任务id。
tasks的每项包含id、question、kind、status、requires、is_example。status仅选answered/needs_user/needs_review/cancelled。
输入seeded任务的kind=other只是规划占位，输出必须按实际问题重新归类；问保费用price，年/月差额用price_difference，实际费用测算用claim_calculation。不要照抄占位kind或空requires。
facts的每项包含name、value、state、user_quote；confirmed的user_quote必须逐字来自实际用户原话，不能以坐席或资料证明用户条件。
报价必须保留age、has_social的已确认依据；其它依赖按本轮责任确定，缺条件就clarify并实际追问。
本要求只是输出协议，不允许补编用户条件或绕过证据、待办与播放检查。'''
            tasks=getattr(self._context,'consultation',{}).get('tasks',[])
            user+='\n【输出前核对本轮任务ID，不是新证据】\n'+json.dumps(tasks,ensure_ascii=False)
            user+='\n本轮用户原话：'+getattr(self._context,'question','')
            user+='\n最终segments每个条目必须有task_ids；处理输入任务时原id同时存在于consultation.tasks和实际答句的task_ids中。请完整输出持续咨询v4的JSON，不要遗漏后台任务及事实字段。'
        return super().call(system,user,self.model)


def clone(session):
    return CallSession(**{f.name:session.pipe if f.name=='pipe' else deepcopy(getattr(session,f.name))
                          for f in fields(session)})


def demo_text(text):
    text=text.replace('请问是给您自己买，还是给家人买？','请问您是为自己了解，还是帮家人了解？')
    for old,new in zip(_sentences(LINK_SENT),(
            '演示环境已记录投保链接请求，短信服务尚未接入。',
            '投保仍需在真实渠道完成健康告知。','本次演示到这里，感谢您的时间。')):
        text=text.replace(old,new)
    for old,new in zip(_sentences(HANDOFF),(
            '我已记录转人工请求，演示环境尚未实际接通人工。',
            '后续仍需通过真实客服渠道核实。')):
        text=text.replace(old,new)
    return text


def fixed_speech():
    from agent.fast_lane import FAST_PHRASES
    from agent.session import ASK, ASK_LINK, LIMITS, DISCLOSURE, RETAIN, GOODBYE, POST_HANDOFF
    texts=[OUTBOUND_OPENING,PRODUCT_INTRO,PRODUCT_DETAIL,ACK,CLARIFY,'这部分已经介绍完了，您想具体了解哪一点？',
           '保费主要按被保人的年龄和医保情况核对。请问被保人今年多少周岁？',
           '请问被保人有基本医保吗？',
           '请问您是首次投保必选计划个人单，还是续保、重新投保或其他计划？',
           *FAST_PHRASES,*[demo_text(t) for t in ASK.values()],LIMITS,DISCLOSURE,ASK_LINK,RETAIN,
           GOODBYE,POST_HANDOFF,demo_text(LINK_SENT),demo_text(HANDOFF)]
    return list(dict.fromkeys(sentence for text in texts for sentence in _sentences(text)))


def wav_base64(pcm):
    out=io.BytesIO()
    with wave.open(out,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(pcm)
    return base64.b64encode(out.getvalue()).decode('ascii')


@dataclass
class Speech:
    text: str
    received: bool = False
    kind: str = 'flow'


class DemoSession:
    def __init__(self,mode='replay',provider='deepseek',pipe=None,scenario='consultation',speech_cache=None):
        self.id=uuid.uuid4().hex
        self.mode,self.provider=mode,provider
        self.scenario=scenario
        self.opening_started=False
        self.outbound_step=0
        self._outbound_next_step=None
        self.speech_cache=speech_cache or SpeechCache()
        self.pipe=pipe
        self.session=CallSession(pipe,voice=True,playback_tracking=True) if pipe is not None else None
        self.replay_index=0
        self.replay_state={}
        self.turn_id=0
        self.cancelled=threading.Event()
        self.speech=[]
        self.active_text=''
        self.result=None
        self.task=None
        self.connected=False
        self.last_seen=time.monotonic()
        self.log=[]
        self.voice_events=[]
        self.tts=None
        self.tts_lock=threading.Lock()
        self._generation_committed=False
        self.timeline=None
        self.current_record=None
        self.turn_timeout_s=18

    def replay_row(self,index=None):
        index=self.replay_index if index is None else index
        path=REPLAY/f'{index+1:02d}_turn.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

    def heard(self):
        return ''.join(s.text for s in self.speech if s.received)

    def snapshot(self):
        self.last_seen=time.monotonic()
        if self.session is not None:
            s=self.session
            state={'history':s.history,'profile':s.profile,'consultation':s.consultation,
                   'handoff_request':s.handoff_request,'ended':s.ended,'stage':s.stage}
        else:
            state=deepcopy(self.replay_state) or {'history':[],'profile':{},'consultation':{},'ended':False,'stage':'准备回放'}
        next_row=self.replay_row() if self.mode=='replay' else None
        return {**deepcopy(state),'id':self.id,'mode':self.mode,'provider':self.provider,
                'scenario':self.scenario,'opening_started':self.opening_started,
                'outbound_step':self.outbound_step,
                'next_question':next_row['question'] if next_row else None,
                'replay_step':self.replay_index,'turn_id':self.turn_id,
                'played_text':self.heard(),'core_tag':CORE_TAG,'demo_tag':DEMO_TAG,
                'turns':deepcopy(self.log),'voice_events':deepcopy(self.voice_events[-64:])}

    def receipt(self,turn,index):
        if turn!=self.turn_id or self.cancelled.is_set():
            return False
        # Browser reports completed sentences in order; arbitrary text is never accepted.
        expected=sum(s.received for s in self.speech)
        if type(index) is not int or index!=expected or index>=len(self.speech):
            return False
        self.speech[index].received=True
        self._commit_receipt()
        if self.current_record:
            self.current_record['heard']=self.heard()
            self.current_record['playback_complete']=all(s.received for s in self.speech) and self._generation_committed
        return True

    def browser_metric(self, turn, data):
        if turn!=self.turn_id or not self.current_record or self.cancelled.is_set():return False
        import math
        allowed=('first_sound_s','first_business_sound_s','playback_end_s','recording_s','asr_connect_s','asr_finalize_s','asr_endpoint_s','rate')
        valid={k:float(v) for k,v in data.items() if k in allowed and type(v) in (int,float)
               and math.isfinite(v) and 0<=v<=600}
        self.current_record.setdefault('browser_metrics',{}).update(valid)
        return True

    def _commit_receipt(self):
        if self.result is not None and self.session is not None:
            heard=self.heard()
            self.session.confirm_playback(self.result.playback_id,heard,
                complete=heard==self.result.text)
            if heard==self.result.text and self._outbound_next_step is not None:
                self.outbound_step=self._outbound_next_step
                self._outbound_next_step=None

    def stop(self):
        if self.cancelled.is_set():return
        self.cancelled.set()
        if self.session is not None:
            if self.result is not None:
                self.session.interrupt_playback(self.result.playback_id)
            elif not self._generation_committed and (self.active_text or self.heard()):
                # Discard pending clone mutations, retaining actual user input and completed speech.
                if self.active_text:self.session.history.append(('用户',self.active_text))
                if self.heard():self.session.history.append(('坐席',self.heard()))
        if self.current_record and not self.current_record.get('playback_complete'):
            self.current_record.update(heard=self.heard(),cancelled=True)
            if self.timeline:self.current_record['stages']=self.timeline.snapshot()

    def export(self):
        return {'session':self.snapshot(),'turns':deepcopy(self.log),'voice_events':deepcopy(self.voice_events),
                'note':'回放为exp44历史记录；浏览器播放回执不是声学DAC验收。' }

    def voice_event(self,turn,kind,text='',metrics=None):
        allowed={'microphone_on','microphone_off','speech_pause','confirmed_interrupt','backchannel','echo','stop_only','asr_renew'}
        if not isinstance(kind,str) or kind not in allowed or type(turn) is not int or turn!=self.turn_id:return False
        if not isinstance(text,str) or len(text)>2000:return False
        import math
        clean={k:float(v) for k,v in (metrics or {}).items() if k in ('vad_pause_s','barge_confirm_s')
               and type(v) in (int,float) and math.isfinite(v) and 0<=v<=600}
        self.voice_events.append({'turn':turn,'kind':kind,'text':text,'metrics':clean,
                                  'source':'browser','time':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        self.voice_events=self.voice_events[-512:]
        return True

    def synthesize(self,text,check,timeline,index):
        check()
        with timeline.span('语音缓存读取',category='tts',sentence=index) as cached_stage:
            cached=self.speech_cache.get(text)
            cached_stage['cache_hit']=cached is not None
        if cached is not None:
            check()
            return cached
        if text in fixed_speech():
            # Never make an opening wait on a cloud request. Preparation runs separately.
            raise SpeechNotPreparedError('Fixed speech is not prepared')
        # A cancelled asyncio task cannot stop a blocking synth call. Serialize
        # connection use so its late audio never mixes with the next question.
        with self.tts_lock:
            check()
            if self.tts is None:
                with timeline.span('语音服务连接',category='tts'):
                    self.tts=WebRealtimeTTS()
            check()
            with timeline.span('语音合成',category='tts',sentence=index) as stage:
                first,total,pcm=self.tts.synth(text)
                stage.update(first_packet_s=first,audio_duration_s=len(pcm)/48000)
            check()
            return pcm

    def close_tts(self):
        with self.tts_lock:
            if self.tts is not None:self.tts.close();self.tts=None

    async def run(self,text,send,cloud_voice=False,opening=False):
        if self.session is not None and self.session.ended:
            raise ValueError('会话已结束，请新建会话。')
        if opening:
            if self.mode!='live' or self.scenario!='outbound' or self.session is None:
                raise ValueError('请在实时主动推销场景中开始演示。')
            if self.opening_started or self.turn_id:
                raise ValueError('本会话已开始，请继续对话或新建会话。')
            self.opening_started=True
            text=''
        elif self.scenario=='outbound' and not self.opening_started:
            raise ValueError('请先点击“开始推销演示”，由AI主动开场。')
        row=self.replay_row() if self.mode=='replay' else None
        if self.mode=='replay' and (row is None or text!=row['question']):
            raise ValueError('回放只能依次播放已录制问题；自由提问请选择实时咨询。')
        self.turn_id+=1
        epoch=self.turn_id
        cancelled=self.cancelled=threading.Event()
        self.speech=[];self.active_text=text;self.result=None;self._generation_committed=False
        self._outbound_next_step=None
        self.current_record={'turn':epoch,'question':text,'response':'','heard':'','cancelled':False,
                             'kind':'opening' if opening else 'turn','scenario':self.scenario,
                             'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'mode':self.mode,
                             'provider':self.provider,'browser_metrics':{},'stages':[]}
        self.log.append(self.current_record)
        await send({'type':'turn_start','turn':epoch,'question':text,'kind':'opening' if opening else 'turn'})
        begin=time.perf_counter()
        loop=asyncio.get_running_loop()
        queue=asyncio.Queue()
        candidate=clone(self.session) if self.session is not None else None
        outbound_next_step=None
        timeline=self.timeline=Timeline(lambda entry:loop.call_soon_threadsafe(queue.put_nowait,{'stage':entry}))
        if candidate is not None and isinstance(self.pipe,EvidencePipeline):
            candidate.pipe=ObservedPipeline(self.pipe,timeline,WebAnswerer)

        def check():
            if cancelled.is_set():raise RuntimeError('cancelled')

        def emit(piece):
            check()
            for sentence in _sentences(demo_text(piece) if candidate is not None else piece):
                kind='business' if timeline.business_ready else 'acknowledgment' if candidate is not None else 'replay'
                loop.call_soon_threadsafe(queue.put_nowait,{'sentence':sentence,'kind':kind})

        def generate():
            nonlocal outbound_next_step
            try:
                check()
                if candidate is not None:
                    if opening:
                        with timeline.span('主动推销开场',category='phase'):
                            candidate.opening()
                            speech=OUTBOUND_OPENING
                            # This browser demo has playback receipts, not a call recorder.
                            candidate.flags['recording']=False
                            candidate._playback_records[candidate._playback_active]['generated']=speech
                            emit(speech)
                        return TurnResult(speech,candidate.stage,'主动开场',['主动推销:开场'],
                                          playback_id=candidate._playback_active)
                    decision={'tool':'answer_question'}
                    if self.scenario=='outbound':
                        with timeline.span('小模型选择会话工具',category='phase') as dispatch_stage:
                            decision=select_action(text,candidate,self.log[:-1],self.outbound_step,candidate.pipe.client)
                            dispatch_stage['tool']=decision['tool']
                        check()
                        self.current_record['dispatch']=decision
                    tool=decision['tool']
                    if tool=='collect_quote_conditions':
                        if not re.search(r'如果|假如|假设|要是|万一|比有|比没',text):
                            candidate._update_profile(extract_slots(text))
                        user_texts=[t for speaker,t in candidate.history if speaker=='用户']+[text]
                        if candidate.profile.get('age') is None:
                            speech='保费主要按被保人的年龄和医保情况核对。请问被保人今年多少周岁？'
                        elif candidate.profile.get('has_social') is None:
                            speech='请问被保人有基本医保吗？'
                        elif not consult.quote_scope_known(candidate.profile,user_texts):
                            speech='请问您是首次投保必选计划个人单，还是续保、重新投保或其他计划？'
                        else:tool='answer_question'
                        if tool=='collect_quote_conditions':
                            candidate.stage='报价条件'
                            if not consult.pending(candidate.consultation):
                                candidate.consultation=consult.start(text,f'web-{epoch}')
                            self.session.consultation=deepcopy(candidate.consultation)
                    if tool!='answer_question':
                        if tool=='collect_quote_conditions':pass
                        elif tool=='play_script':
                            script_id=decision['script_id']
                            speech=script_remainder(script_id,self.log[:-1],decision.get('restart',False))
                            if speech:
                                self.current_record['script_id']=script_id
                                candidate.stage='产品介绍' if script_id=='product_intro' else '保障介绍'
                                outbound_next_step=max(self.outbound_step,1 if script_id=='product_intro' else 2)
                            else:speech='这部分已经介绍完了，您想具体了解哪一点？'
                        elif tool=='end_call':
                            speech='好的，不打扰您了，祝您生活愉快，再见。'
                            candidate.stage='结束';candidate.ended=True
                        elif tool=='recall_question':
                            index=decision.get('question_index')
                            speech=('您刚才说的是：“'+candidate.history[index][1]+'”'
                                    if index is not None else '您刚才是在让我继续介绍产品，还没有提出具体的保险问题。')
                        else:speech=ACK if tool=='acknowledge' else CLARIFY
                        candidate.history.append(('用户',text))
                        candidate._begin_playback()
                        timeline.business_ready=True
                        candidate._say(speech)
                        emit(speech)
                        return TurnResult(speech,candidate.stage,tool,
                                          ['主动推销:小模型调度:'+tool],
                                          playback_id=candidate._playback_active)
                    if self.scenario=='outbound' and not consult.pending(self.session.consultation):
                        self.session.consultation=consult.start(text,f'web-pending-{epoch}')
                    return candidate.turn(text,on_sentence=emit,cancel_check=check)
                for piece in _sentences(row['session_text']):emit(piece)
                return None
            finally:
                timeline.text_ready_s=timeline.elapsed()
                loop.call_soon_threadsafe(queue.put_nowait,None)

        future=loop.run_in_executor(None,generate)
        try:
            while True:
                remaining=self.turn_timeout_s-(time.perf_counter()-begin)
                if remaining<=0 and not future.done():raise asyncio.TimeoutError
                piece=(await asyncio.wait_for(queue.get(),max(remaining,0.01))
                       if not future.done() else await queue.get())
                if piece is None:break
                check()
                if epoch!=self.turn_id:return
                if 'stage' in piece:
                    await send({'type':'stage','turn':epoch,'stage':piece['stage']});continue
                kind=piece['kind'];piece=piece['sentence']
                self.current_record['response']+=piece
                index=len(self.speech);self.speech.append(Speech(piece,kind=kind))
                await send({'type':'sentence','turn':epoch,'index':index,'text':piece,
                            'kind':kind,'elapsed_s':round(time.perf_counter()-begin,3)})
                if cloud_voice and self.mode=='live':
                    try:
                        pcm=await asyncio.to_thread(self.synthesize,piece,check,timeline,index)
                        check()
                        await send({'type':'audio','turn':epoch,'index':index,'wav':wav_base64(pcm)})
                    except Exception as exc:
                        if cancelled.is_set():break
                        await send({'type':'audio_fallback','turn':epoch,'index':index,
                                    'message':('固定话术音频正在准备，本句先用浏览器朗读。' if isinstance(exc,SpeechNotPreparedError)
                                               else '云语音暂不可用，此句可用浏览器朗读。')})
                        if self.tts is not None:
                            await asyncio.to_thread(self.close_tts)
                else:
                    await send({'type':'audio_fallback','turn':epoch,'index':index,'message':''})
            result=await future
            check()
            if epoch!=self.turn_id:return
            if candidate is not None:
                safe_text=demo_text(result.text)
                if safe_text!=result.text:
                    result.text=safe_text
                    candidate._playback_records[result.playback_id]['generated']=safe_text
                    candidate.flags['link']=False
                    result.events.append('演示适配:未集成动作只记录请求')
                self.session=candidate;self.result=result;self._generation_committed=True
                self._outbound_next_step=outbound_next_step
                self._commit_receipt()
                detail=result.timings.get('answer_detail',{}).get('evidence_result',{})
                citations=[]
                for claim in detail.get('structured',{}).get('claims',[]):
                    for citation in claim.get('citations',[]):
                        if not any(c['id']==citation['id'] for c in citations):
                            citations.append({'id':citation['id'],'quote':citation.get('quote',''),
                                              'location':citation.get('location',{})})
                trace={'events':result.events,'citations':citations,'timings':{
                    k:v for k,v in result.timings.items() if k in ('total','understand','answer')},
                       'validation':result.timings.get('answer_validation',{}),
                       'claims':detail.get('structured',{}).get('claims',[]),
                       'consultation_fault':result.timings.get('answer_detail',{}).get('consultation_fault')}
                response=result.text
            else:
                self.replay_state={k:deepcopy(row.get(k)) for k in ('profile','consultation','history_after','handoff_request','ended')}
                self.replay_state['history']=self.replay_state.pop('history_after')
                self.replay_state['stage']='历史记录回放'
                self.replay_index+=1
                detail=row['timings'].get('answer_detail',{}).get('evidence_result',{})
                citations=[]
                for claim in detail.get('structured',{}).get('claims',[]):
                    citations.extend({'id':c['id'],'quote':c.get('quote',''),'location':c.get('location',{})}
                                     for c in claim.get('citations',[]))
                trace={'events':row.get('events',[]),'citations':citations,
                       'recorded_playback_end_s':row.get('actual_dac_playback_end_s'),
                       'recorded_case':row['id'],'recorded_batch':'exp44_voice_current'}
                trace['claims']=detail.get('structured',{}).get('claims',[])
                response=row['session_text']
            trace.update(stages=timeline.snapshot(),text_ready_s=timeline.text_ready_s,
                         audio_ready_s=round(time.perf_counter()-begin,3))
            self._generation_committed=True
            self.current_record.update(response=response,trace=trace,stages=timeline.snapshot(),
                                       heard=self.heard(),playback_complete=all(s.received for s in self.speech))
            await send({'type':'turn_done','turn':epoch,'text':response,'trace':trace,
                        'elapsed_s':round(time.perf_counter()-begin,3),'state':self.snapshot()})
        except asyncio.TimeoutError:
            self.stop()
            future.add_done_callback(lambda f:f.exception() if not f.cancelled() else None)
            await send({'type':'error','message':'本轮核对超时，已暂停。您的问题已保留，可以补充条件或重试。',
                        'state':self.snapshot()})
        except asyncio.CancelledError:
            cancelled.set()
            # Observe background completion without releasing its result or leaking an exception.
            future.add_done_callback(lambda f:f.exception() if not f.cancelled() else None)
            raise
        except Exception:
            if not cancelled.is_set():
                self.stop()
                await send({'type':'error','message':'本轮服务未完成，已暂停。请重试或新建会话。',
                            'state':self.snapshot()})

    async def close(self):
        self.stop()
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task,return_exceptions=True)
        if self.tts is not None:
            await asyncio.to_thread(self.close_tts)


class DemoStore:
    def __init__(self):
        self.sessions={}
        self.evidence=None
        self.speech_cache=SpeechCache()

    def warm_speech(self):
        self.speech_cache.warm(fixed_speech())

    def create(self,mode,provider,scenario='consultation'):
        if scenario not in ('consultation','outbound'):
            raise ValueError('不支持的业务场景。')
        if mode=='replay' and scenario=='outbound':
            raise ValueError('主动推销演示需要选择实时模型，历史回放仅支持咨询。')
        now=time.monotonic()
        expired=[key for key,s in self.sessions.items() if not s.connected and now-s.last_seen>3600]
        for key in expired:self.sessions.pop(key)
        if len(self.sessions)>=64:raise ValueError('演示会话数量已满，请关闭旧会话后再试。')
        pipe=None
        if mode=='live':
            if not capabilities().get(provider):raise ValueError('所选模型未配置，请选择录制回放或配置凭据。')
            if self.evidence is None:
                from demo.source_compat import install_source_compat
                install_source_compat(ROOT)
                self.evidence=grouped_evidence()
            client=cloud.qwen_client()
            model=cloud.QWEN_ANSWER_MODEL
            if provider=='deepseek':
                from agent.model_router import AnswerModelRouter
                model=cloud.SLOW_MODEL if cloud.SLOW_MODEL.startswith('deepseek-') else 'deepseek-flash'
                client=AnswerModelRouter(client,model)
            if os.name=='nt' and os.getenv('DEMO_NATIVE_HTTPS')=='1':
                import sys
                import httpx
                sys.path.insert(0,str(ROOT/'experiments'))
                from windows_https_transport import WindowsHttpsTransport
                client=client.with_options(http_client=httpx.Client(transport=WindowsHttpsTransport(total_timeout=12)),
                                           timeout=12,max_retries=0)
            pipe=EvidencePipeline(self.evidence,client=client,answerer=WebAnswerer(client,model))
        session=DemoSession(mode,provider,pipe,scenario,self.speech_cache)
        self.sessions[session.id]=session
        return session
