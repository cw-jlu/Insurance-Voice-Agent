"""Prepared fixed speech and a bounded, handshake-aware TTS connection for the web demo."""
import base64
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

from agent import cloud

OUTBOUND_OPENING = ('您好，我是众安保险的AI助理。今天联系您，是想介绍一下尊享e生百万医疗保险。'
                    '请问您现在方便听我简单介绍一下吗？')
PRODUCT_INTRO = ('好的。平时小病花费还能承担，大家更担心的是万一生场大病，住院、治疗开销大，影响一家人的生活安排。'
                 '尊享e生这款百万医疗险，就是给医保之外符合约定的医疗费用再加一层保障。'
                 '必选计划国内医疗总保额600万元，除了住院，还包含符合条件的特定药品、质子重离子等保障。'
                 '当然，600万是保额上限，实际报销要看费用是否符合保障范围、免赔额等条件。'
                 '我接着给您讲讲，普通住院和大病治疗分别怎么保障。')
PRODUCT_DETAIL = ('普通住院方面，符合约定的一般医疗费用，年度免赔额是1万元，具体赔付还要看医保结算等条件。'
                  '大病治疗方面，符合约定的重大疾病医疗没有免赔额；特定药品、质子重离子也有各自的保障条件和就医要求。'
                  '这份保障的作用，是在发生符合约定的大额医疗支出时，帮助减轻自己和家人的负担。'
                  '同时要提醒您，首年疾病等待期为30天，既往症等免责情形要特别留意，能否投保以健康告知和核保结果为准。'
                  '您可以接着问保障细节；如果想了解保费，我再根据被保人的年龄和医保情况帮您核对。')


class SpeechServiceError(RuntimeError):
    def __init__(self, code):
        self.code=str(code)
        super().__init__('Speech service failed: '+self.code)


class SpeechNotPreparedError(RuntimeError):
    pass


class WebRealtimeTTS:
    """Do not send configuration or text before the server acknowledges readiness."""
    def __init__(self):
        from websockets.sync.client import connect
        url=os.environ['DASHSCOPE_WS_URL'].rsplit('/',1)[0]+'/realtime?model='+cloud.TTS_MODEL
        self.ws=connect(url,additional_headers={'Authorization':'Bearer '+os.environ['DASHSCOPE_API_KEY']},
                        proxy=None,open_timeout=5,close_timeout=1)
        try:
            self.wait_for('session.created',5)
            self.send('session.update',session={'voice':cloud.TTS_VOICE,'mode':'commit',
                                               'response_format':'pcm','sample_rate':24000})
            self.wait_for('session.updated',5)
        except BaseException:
            self.close()
            raise

    def send(self,kind,**data):
        self.ws.send(json.dumps({'event_id':uuid.uuid4().hex,'type':kind,**data}))

    def receive(self,timeout):
        value=json.loads(self.ws.recv(timeout=timeout))
        if value.get('type')=='error':
            raise SpeechServiceError(value.get('error',{}).get('code','unknown'))
        return value

    def wait_for(self,kind,timeout):
        deadline=time.monotonic()+timeout
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('TTS readiness timeout')
            value=self.receive(remaining)
            if value.get('type')==kind:return

    def synth(self,text,timeout=12):
        start=time.monotonic();first=None;audio=bytearray()
        self.send('input_text_buffer.append',text=text)
        self.send('input_text_buffer.commit')
        while True:
            remaining=timeout-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('TTS synthesis timeout')
            value=self.receive(remaining)
            if value.get('type')=='response.audio.delta':
                if first is None:first=time.monotonic()-start
                audio.extend(base64.b64decode(value['delta']))
            if value.get('type')=='response.done':
                if not audio:raise SpeechServiceError('empty_audio')
                return first,time.monotonic()-start,bytes(audio)

    def close(self):
        self.ws.close()


class SpeechCache:
    def __init__(self,directory=None):
        self.directory=Path(directory) if directory is not None else Path(__file__).resolve().parents[2]/'demo-runtime/speech'
        self.audio={}
        self.warm_lock=threading.Lock()
        self.status='idle'
        self.error=None

    def key(self,text):
        value=json.dumps([cloud.TTS_MODEL,cloud.TTS_VOICE,'pcm-24000-mono-int16',text],ensure_ascii=False)
        return hashlib.sha256(value.encode()).hexdigest()

    def get(self,text):
        key=self.key(text)
        if key in self.audio:return self.audio[key]
        try:pcm=(self.directory/(key+'.pcm')).read_bytes()
        except FileNotFoundError:return None
        if not pcm or len(pcm)%2:return None
        self.audio[key]=pcm
        return pcm

    def put(self,text,pcm):
        if not pcm or len(pcm)%2:raise ValueError('Expected nonempty int16 PCM')
        self.directory.mkdir(parents=True,exist_ok=True)
        key=self.key(text);path=self.directory/(key+'.pcm')
        temporary=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
        try:
            temporary.write_bytes(pcm)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        self.audio[key]=pcm

    def warm(self,phrases,factory=WebRealtimeTTS):
        with self.warm_lock:
            self.status='warming';self.error=None;tts=None
            try:
                for text in dict.fromkeys(phrases):
                    if self.get(text) is not None:continue
                    if tts is None:tts=factory()
                    self.put(text,tts.synth(text)[2])
                self.status='ready'
            except Exception as exc:
                self.status='unavailable'
                self.error=getattr(exc,'code',type(exc).__name__)
            finally:
                if tts is not None:
                    try:tts.close()
                    except Exception:pass
