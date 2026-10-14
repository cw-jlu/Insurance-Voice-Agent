"""Bounded lightweight speech; no insurance facts or invented action status."""
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, wait

from agent import cloud
from agent.context import with_history

QUERY_ACK = '我帮您查一下。'
HANDOFF_ACK = '好的，我为您记录转人工请求。'
CORRECTION_ACK = '好的，我按您刚才更正的信息核对。'


def explicit_handoff(text):
    return bool(re.search(r'转人工|人工客服|找人工|要人工|真人客服|转接人工|转专员', text)
                and not re.search(r'不用转|先别转|不要转|别转|不需要人工|不要人工|不想.{0,3}(?:转|找|要).{0,3}人工', text))


def rule_ack(text):
    # Reject hypothetical transfer requests and quoted speech before acting.
    if explicit_handoff(text) and not re.search(r'如果|假如|要是|他说|她说|什么意思|怎么转|怎么找|有没有|是否|你是|是人工', text):
        return 'handoff', HANDOFF_ACK
    if re.search(r'不需要|不买|不要了|别打|不再来电|不想了解|不用介绍|已经了解|了解了|你是.*(?:AI|人工|机器人)', text, re.I):
        return None, ''
    if re.search(r'不对|改成|说错|更正', text) and re.search(r'多少|怎么|吗|呢|？|\?', text):
        return 'correction', CORRECTION_ACK
    if re.search(r'多少|多久|怎么|什么|能不能|是否|有没有|查一下|查查|想了解|了解.{0,12}(?:保险|保障|责任|保费)|介绍一下|讲讲|？|\?', text):
        return 'query', QUERY_ACK
    return None, ''


def model_ack(act):
    # Small-model act chooses a fixed phrase; it cannot add an insurance claim.
    return QUERY_ACK if act in ('提问', '异议') else ''


FAST_PHRASES = (QUERY_ACK, HANDOFF_ACK, CORRECTION_ACK)

WAIT_SYSTEM = '''你是保险语音助手的等待承接小模型。主回答模型正在查证，当前目标是用一句围绕用户问题的自然承接填补等待，给主回答留出时间。
可以接住用户的问意，表示会把并列问题一起核对；不能提前回答保费、承保、赔付、优惠或条款结论，也不能宣布已经回答、已经转接或已发送链接。
只从allowed_focuses选择最多两个用户正在关心的主题，不能增加用户未问的问题。不要提“拖时间”、后台模型或内部流程。用户着急时可选patience，否则选ack。
只输出JSON：{"focuses":["allowed_focuses中的键"],"tone":"ack或patience"}。最终程序按这些主题生成短句，不输出其他话术。'''
WAIT_FOCUSES = {
    'price':('保费',r'多少钱|保费|报价|年缴|月缴|价格'),
    'discount':('优惠',r'优惠|折扣|打折'),
    'eligibility':('投保条件',r'投保|能买|可以保|能买吗|核保'),
    'claim':('赔付',r'能赔|赔多少|赔付|报销|能报|怎么赔'),
    'deductible':('免赔',r'免赔'),
    'waiting':('等待期',r'等待|等.*天'),
    'hospital':('医院范围',r'医院|药店|渠道'),
    'coverage':('保障范围',r'保障|责任|加油包|癌症'),
}


def waiting_reply(question,history,consultation,client):
    """The small model selects grounded topics; it cannot release business facts."""
    task_text='；'.join(t['question'] for t in consultation.get('tasks',[]) if t['status']!='cancelled')
    text=question+'；'+task_text
    allowed=[key for key,(_,pattern) in WAIT_FOCUSES.items() if re.search(pattern,text)]
    shown={'current_user':question,'original_consultation':consultation.get('initial_question'),
           'allowed_focuses':allowed,'latest_message_only_is_not_a_new_consultation':bool(consultation)}
    user='【已播对话】\n'+history+'\n【当前等待】\n'+json.dumps(shown,ensure_ascii=False)
    try:
        raw=cloud.chat(with_history([{'role':'system','content':WAIT_SYSTEM},{'role':'user','content':user}],
                                    history,marker='【已播对话】\n'),
                       model=cloud.FAST_MODEL,client=client.with_options(timeout=2.0,max_retries=0),
                       temperature=0,response_format={'type':'json_object'},max_tokens=100)
        selected=json.loads(raw)
        focuses=selected.get('focuses',[])
        if not isinstance(focuses,list) or len(focuses)>2 or len(set(focuses))!=len(focuses) or any(k not in allowed for k in focuses):
            raise ValueError('Invalid waiting topics')
        tone=selected.get('tone')
        if tone not in ('ack','patience'):raise ValueError('Invalid waiting tone')
        labels='和'.join(WAIT_FOCUSES[k][0] for k in focuses)
        spoken=('您关心的'+labels+'，我一起核对。') if labels else '您刚才的问题，我正在核对。'
        if tone=='patience':spoken='让您久等了，'+spoken
        return {'text':spoken,'focuses':focuses,'tone':tone,'source':'small_model_topic_reply'}
    except Exception as exc:
        return {'text':'','source':'waiting_reply_unavailable','error_type':type(exc).__name__}


def answer_while_waiting(answer_fn,reply_fn,emit,cancel_check=None,delay_s=1.5):
    """Release at most one bridge, on the caller thread, only while DS is pending.

    Cancellation/DS completion takes precedence over a ready bridge. Stale worker
    results never mutate the session or release speech. No repeated filler loop.
    """
    started=time.perf_counter()
    pool=ThreadPoolExecutor(2)
    main=pool.submit(answer_fn)
    bridge=None
    info={'emitted':False,'max_replies':1,'delay_s':delay_s}
    try:
        while True:
            if cancel_check is not None:cancel_check()
            if main.done():
                result=main.result()
                info['answer_ready_s']=time.perf_counter()-started
                if bridge is not None and not info['emitted']:
                    info['suppressed_on_answer_ready']=True
                return result,info
            if bridge is None and time.perf_counter()-started>=delay_s:
                bridge=pool.submit(reply_fn)
            if bridge is not None and bridge.done() and not info.get('reply_received'):
                candidate=bridge.result()
                info.update(candidate,reply_received=True)
                if cancel_check is not None:cancel_check()
                if not main.done() and candidate['text']:
                    emit(candidate['text'])
                    info['emitted']=True
                    info['emit_s']=time.perf_counter()-started
            wait([main]+([bridge] if bridge is not None and not bridge.done() else []),timeout=.02)
    finally:
        main.cancel()
        if bridge is not None:bridge.cancel()
        pool.shutdown(wait=False,cancel_futures=True)


class CachedSpeech:
    """Prewarm fixed phrases before accepting input; use ordinary TTS otherwise."""
    def __init__(self, tts):
        self.tts = tts
        self.audio = {}

    def warm(self):
        for text in FAST_PHRASES:
            self.audio[text] = self.tts.synth(text)[2]

    def synth(self, text, on_audio=None):
        if text not in self.audio:
            return self.tts.synth(text, on_audio=on_audio)
        audio = self.audio[text]
        if on_audio is not None:
            on_audio(audio)
        return 0.0, 0.0, audio
