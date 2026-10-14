"""Frozen evidence v2 + live ASR/session/TTS/WASAPI playback acceptance.

Synthetic input/echo injection, real cloud services and output device timing.
No physical microphone, telephone, or acoustic loopback recording is claimed.
"""
import argparse
import asyncio
import hashlib
import json
import math
import queue
import statistics
import sys
import threading
import time
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from agent import cloud, small_model
from agent.evidence_answer import Evidence, handoff_system
from agent.evidence_pipeline import EvidencePipeline
from agent.session import CallSession
from agent.dialogue import BargeInController
from exp4_e2e import speech_end_s
from exp5_audio import mix, first_voiced_s
from exp30_fewshot import save

OUT=ROOT/'experiments/results'
REC=ROOT/'experiments/recordings/exp36'
IDS=['C01','C02','C09','C19','C26','C10','E13','E25','H04','H07','H10','H15']
INTERRUPTS=['等一下，我没有社保','不对，我今年40岁','不是，我是给我妈买的','先停一下',
    '门诊能报吗','等待期多久','那按月交多少','我不需要了','你是机器人吗','转人工',
    '我没有医保，是30岁，年缴多少钱','我今年40岁有社保，一年多少钱',
    '不是，我女儿是5岁','我想按月交','意外也要等30天吗','我已经有保险了',
    '免赔额是多少','我有甲状腺结节能买吗','别给我打电话了','你先听我说']
BACKCHANNELS=['嗯','对','好的','明白','哦','嗯嗯，你说','好，你继续','了解','收到','对的']
GAINS=[.03,.10,.30]

def freeze():
    previous=json.loads((OUT/'exp35_plan.json').read_text(encoding='utf-8'))
    cases=[c for c in previous['cases'] if c['id'] in IDS]
    # One spoken turn; no prices or audit labels supplied to answering model.
    for c in cases:
        if c['id']=='H15':
            c['question']='我30岁有社保首次买必选计划年缴多少钱以及原位癌能否按这份合同的100种重疾报销请确认定义'
    paths=['src/agent/evidence_answer.py','src/agent/evidence_pipeline.py','src/agent/session.py',
        'src/agent/cloud.py','src/agent/dialogue.py','src/agent/small_model.py','src/agent/pipeline.py']
    plan={'date':'2026-10-12','model':cloud.SLOW_MODEL,'fast_model':cloud.FAST_MODEL,
        'evidence_sha256':hashlib.sha256((OUT/'exp33_organized_sources.json').read_bytes()).hexdigest(),
        'prompt_sha256':hashlib.sha256(handoff_system('v2').encode()).hexdigest(),
        'code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        'voice_cases':cases,'rounds':2,'vad_ms':700,'tts_model':cloud.TTS_MODEL,'speech_rate':1.2,
        'playback_device':dict(sd.query_devices(8)),'output_sample_rate':48000,
        'interrupts':INTERRUPTS,'backchannels':BACKCHANNELS,'echo_gains':GAINS,'echo_only_per_gain':8,
        'method':'24 fresh live voice turns with unchanged evidence handoff v2; no model grader. Real streamed input ASR, real flow understanding/classification, fresh DS output, real TTS packets and WASAPI DAC timestamps. Injected synthetic echo tests decisions during live ASR, not offline replay. No microphone/acoustic/telephone validation.'}
    path=OUT/'exp36_plan.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==plan,'Frozen plan changed'
    else:save(path,plan)
    return plan

class DacPlayer:
    def __init__(self,device=None):
        from agent.playback import output_device,output_sample_rate
        device=output_device(sd) if device is None else device
        self.output_rate=output_sample_rate(sd,device)
        self.lock=threading.Lock();self.chunks=deque();self.offset=0
        self.first_dac=None;self.last_dac=None;self.aborted=False
        self.queue_empty=threading.Event();self.queue_empty.set()
        self.statuses=[];self.dropped_frames=0
        def callback(outdata,frames,info,status):
            buf=np.frombuffer(outdata,dtype=np.int16).reshape(-1,2);buf[:]=0
            now=time.perf_counter()
            dac=now+info.outputBufferDacTime-info.currentTime
            with self.lock:
                if status:self.statuses.append(str(status))
                pos=0
                while self.chunks and pos<frames and not self.aborted:
                    a=self.chunks[0];n=min(frames-pos,len(a)-self.offset)
                    buf[pos:pos+n,0]=a[self.offset:self.offset+n]
                    buf[pos:pos+n,1]=a[self.offset:self.offset+n]
                    if self.first_dac is None:
                        audible=np.flatnonzero(np.abs(buf[pos:pos+n,0].astype(np.int32))>327)
                        if len(audible):self.first_dac=dac+(pos+int(audible[0]))/self.output_rate
                    self.last_dac=dac+(pos+n)/self.output_rate
                    pos+=n;self.offset+=n
                    if self.offset==len(a):self.chunks.popleft();self.offset=0
                if not self.chunks:self.queue_empty.set()
        self.stream=sd.RawOutputStream(device=device,samplerate=self.output_rate,channels=2,dtype='int16',
            blocksize=round(self.output_rate*.02),latency='low',callback=callback)
        self.stream.start()
    def feed(self,pcm24):
        from math import gcd
        a=np.frombuffer(pcm24,dtype=np.int16).astype(np.float32)
        divisor=gcd(self.output_rate,24000)
        data=np.clip(resample_poly(a,self.output_rate//divisor,24000//divisor),-32768,32767).astype(np.int16)
        with self.lock:
            if not self.aborted and len(data):self.chunks.append(data);self.queue_empty.clear()
    def abort(self):
        at=time.perf_counter()
        with self.lock:
            self.aborted=True
            self.dropped_frames=sum(len(x) for x in self.chunks)-self.offset
            self.chunks.clear();self.offset=0;self.queue_empty.set()
            end=max(at,self.last_dac or at)
        return {'request_wall':at,'last_scheduled_dac_wall':end,'dropped_frames':self.dropped_frames,
            'stream_latency_s':self.stream.latency}
    def drain(self,timeout=120):
        if not self.queue_empty.wait(timeout):raise TimeoutError('Playback did not drain')
        end=self.last_dac or time.perf_counter()
        if end>time.perf_counter():time.sleep(end-time.perf_counter())
        return end
    def close(self):
        self.stream.stop();self.stream.close()

def wav(path,pcm,rate=24000):
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes(pcm)

def user_audio(question,voice):
    key=hashlib.sha256((voice+question).encode()).hexdigest()[:18]
    path=REC/(key+'.pcm')
    if not path.exists():
        tts=cloud.RealtimeTTS(voice=voice)
        try:pcm=cloud.resample_24k_to_16k(tts.synth(question)[2])
        finally:tts.close()
        path.write_bytes(pcm)
    return path.read_bytes()

def voice_one(case,round_,evidence):
    row={'id':case['id'],'round':round_,'question':case['question']}
    pcm=user_audio(case['question'],'Ethan' if round_==1 else 'Serena')
    wav(REC/f"input_{case['id']}_r{round_}.wav",pcm,16000)
    session=CallSession(EvidencePipeline(evidence),profile={'relation':'本人',**case['known']})
    session.opening();session.flags.update(quoted=True,limits=True,disclosure=True)
    player=DacPlayer();tts=cloud.RealtimeTTS(voice='Cherry',speech_rate=1.2)
    q=queue.Queue();chunks=[];errors=[];origin=[None];result=[None];task=[None]
    final=[None];text_done=[None];sentences=[]
    def emit(text):q.put((text,time.perf_counter()))
    def worker():
        while True:
            item=q.get()
            if item is None:return
            text,emitted=item;t0=time.perf_counter()
            received=[]
            def packet(audio):
                received.append(time.perf_counter());player.feed(audio)
            try:
                first,done,audio=tts.synth(text,on_audio=packet)
                chunks.append(audio)
                sentences.append({'text':text,'emit_wall':emitted,'tts_start_wall':t0,
                    'first_packet_wall':received[0] if received else None,'tts_first_s':first,
                    'tts_done_s':done,'duration_s':len(audio)/48000})
            except Exception as exc:errors.append(type(exc).__name__)
    thread=threading.Thread(target=worker);thread.start()
    work=ThreadPoolExecutor(1)
    def answer(text):
        result[0]=session.turn(text,on_sentence=emit)
        text_done[0]=time.perf_counter()
        return result[0]
    def event(ev):
        if origin[0] is None:origin[0]=time.perf_counter()-ev.t
        if ev.slice_type==2 and task[0] is None:
            final[0]=ev;task[0]=work.submit(answer,ev.text)
    try:
        events,_=asyncio.run(cloud.tencent_asr_stream(pcm,vad_silence_ms=700,on_event=event))
        if task[0] is None:raise RuntimeError('No ASR final')
        task[0].result();q.put(None);thread.join()
        if errors:raise RuntimeError('TTS worker failed')
        end=player.drain();user_end=origin[0]+speech_end_s(pcm)
        finals=[vars(e) for e in events if e.slice_type==2]
        row.update(asr_text=final[0].text,asr_events=[vars(e) for e in events],
            asr_final_count=len(finals),split_intended_turn=len(finals)!=1,
            session_text=result[0].text,events=result[0].events,flags=session.flags,
            profile=session.profile,timings=result[0].timings,
            text_ready_s=text_done[0]-user_end,
            asr_endpoint_s=origin[0]+final[0].t-user_end,
            first_business_pcm_s=min(s['first_packet_wall'] for s in sentences if s['first_packet_wall'])-user_end,
            first_business_dac_s=player.first_dac-user_end if player.first_dac else None,
            actual_dac_playback_end_s=end-user_end,
            audio_duration_s=sum(s['duration_s'] for s in sentences),
            sentences=[{k:(v-user_end if k.endswith('_wall') and v is not None else v) for k,v in s.items()} for s in sentences],
            playback_statuses=player.statuses,output_latency_s=player.stream.latency,
            recording=str(REC/f"answer_{case['id']}_r{round_}.wav"))
        wav(Path(row['recording']),b''.join(chunks))
    except Exception as exc:
        row['error_type']=type(exc).__name__;player.abort()
    finally:
        work.shutdown(wait=True);q.put(None);thread.join();tts.close();player.close()
    return row

def voice_main(plan):
    evidence=[Evidence(**e) for e in json.loads((OUT/'exp33_organized_sources.json').read_text(encoding='utf-8'))['evidence']]
    folder=OUT/'exp36_voice_records';folder.mkdir(exist_ok=True)
    for round_ in [1,2]:
        for case in plan['voice_cases']:
            path=folder/f"{case['id']}_r{round_}.json"
            if path.exists():continue
            row=voice_one(case,round_,evidence);save(path,row)
            print(json.dumps({k:row.get(k) for k in ['id','round','first_business_dac_s','actual_dac_playback_end_s','split_intended_turn','error_type']},ensure_ascii=False),flush=True)

def interrupt_one(job,agent_audio,client):
    kind,text,gain,index,agent=job
    key=f"g{int(gain*100):02d}_{kind}_{index:02d}"
    row={'key':key,'kind':kind,'user':text,'gain':gain,'agent_record':agent['id'],'agent_round':agent['round']}
    voice='Ethan' if index%2==0 else 'Serena'
    upcm=user_audio(text,voice) if text else None
    agent24=agent_audio[(agent['id'],agent['round'])]
    a16=cloud.resample_24k_to_16k(agent24)
    # Pad output by repeating real spoken audio, not fake insurance text.
    # This ensures enough audible time to test late detection without natural end confounds.
    while len(a16)<16000*2*12:a16+=a16
    a16=a16[:16000*2*12]
    a24=np.clip(resample_poly(np.frombuffer(a16,dtype=np.int16).astype(np.float32),3,2),-32768,32767).astype(np.int16).tobytes()
    at=1.2+(index%4)*.25
    mic,onset,agent_end=mix(a16,upcm,at,gain)
    ctrl=BargeInController(lambda t:small_model.classify_barge_in(t,client=client))
    player=DacPlayer()
    origin=[None];decisions=[];stopped=[None];jobs=[];lock=threading.Lock()
    pool=ThreadPoolExecutor(1)
    ctrl.note_spoken(0,agent['session_text'])
    def decide(ev):
        if stopped[0] is not None:return
        now=time.perf_counter()-origin[0]
        d=ctrl.on_partial(now,ev.text,now<12)
        if d is None:return
        at_done=time.perf_counter()-origin[0]
        decisions.append({'asr':ev.text,'received_s':ev.t,'decision_s':at_done,'action':d.action,'reason':d.reason})
        if d.action=='stop':
            with lock:
                if stopped[0] is None:
                    stop=player.abort();stopped[0]={**stop,'decision_s':at_done,
                        'stop_dac_s':stop['last_scheduled_dac_wall']-origin[0]}
    def event(ev):
        if origin[0] is None:origin[0]=time.perf_counter()-ev.t
        if ev.text:jobs.append(pool.submit(decide,ev))
    def frame(data,audio_t):
        if origin[0] is None:
            origin[0]=time.perf_counter()-(audio_t-len(data)/32000)
            player.feed(a24)
    try:
        events,_=asyncio.run(cloud.tencent_asr_stream(mic,vad_silence_ms=500,
            trailing_silence_ms=0,on_event=event,on_audio=frame))
        for job in jobs:job.result()
        row.update(onset_s=onset,decisions=decisions,asr_events=[vars(e) for e in events],
            stopped=stopped[0] is not None,stop_latency_s=stopped[0]['stop_dac_s']-onset if stopped[0] and onset is not None else None,
            wrong_stop_before_user=bool(stopped[0] and onset is not None and stopped[0]['stop_dac_s']<onset),
            cancellation=stopped[0],playback_statuses=player.statuses,
            stale_generation_tested=False,
            limitation='Live controller and actual output-buffer stop; injected synthetic echo, not physical microphone/AEC. Playback audio is a real new-chain reply repeated to 12 seconds. No in-flight generation or corrected-condition response in this subtest.')
    except Exception as exc:row['error_type']=type(exc).__name__
    finally:
        player.abort();pool.shutdown(wait=True);player.close()
    return row

def interrupt_main(plan):
    folder=OUT/'exp36_interrupt_records';folder.mkdir(exist_ok=True)
    agents=[json.loads(f.read_text(encoding='utf-8')) for f in sorted((OUT/'exp36_voice_records').glob('*.json'))]
    agents=[a for a in agents if not a.get('error_type') and not a.get('split_intended_turn') and a['id'] in ['C01','C02','C09','C19']]
    assert agents,'Need fresh new-chain spoken replies first'
    agent_audio={(a['id'],a['round']):Path(a['recording']).read_bytes()[44:] for a in agents}
    # Read WAV data with wave rather than assuming a fixed header.
    for a in agents:
        with wave.open(a['recording'],'rb') as w:agent_audio[(a['id'],a['round'])]=w.readframes(w.getnframes())
    jobs=[]
    # Acceptance gate starts with the least echo. If this gate fails, stronger
    # echo stress tests cannot make the basic 500 ms / 95% requirement pass.
    for gain in GAINS[:1]:
        for kind,texts in [('interrupt',INTERRUPTS),('backchannel',BACKCHANNELS),('echo',[None]*8)]:
            for i,text in enumerate(texts):jobs.append((kind,text,gain,i,agents[i%len(agents)]))
    # Prebuild inputs serially to avoid concurrent WebSocket initialization and duplicate cache writes.
    for kind,text,gain,i,a in jobs:
        if text:user_audio(text,'Ethan' if i%2==0 else 'Serena')
    client=cloud.llm_client()
    # This Windows device rejected overlapping output streams. Single playback
    # after voice_main completes; same frozen questions and echo conditions.
    for job in jobs:
        key=f"g{int(job[2]*100):02d}_{job[0]}_{job[3]:02d}"
        if (folder/(key+'.json')).exists():continue
        try:row=interrupt_one(job,agent_audio,client)
        except Exception as exc:
            row={'key':key,'kind':job[0],'user':job[1],'gain':job[2],
                'error_type':type(exc).__name__,'stage':'trial_start'}
        save(folder/(key+'.json'),row)
        print(json.dumps({k:row.get(k) for k in ['key','kind','stopped','stop_latency_s','wrong_stop_before_user','error_type']},ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['voice','interrupt'],required=True)
    args=ap.parse_args();REC.mkdir(parents=True,exist_ok=True);plan=freeze()
    (voice_main if args.mode=='voice' else interrupt_main)(plan)

if __name__=='__main__':main()
