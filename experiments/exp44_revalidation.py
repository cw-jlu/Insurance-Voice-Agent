"""Fresh current-chain text, real voice and live injected interruption checks.

Frozen inputs, all raw results retained, no hidden expected labels sent to agent.
Certificates checked using the native HTTPS test transport; no network rewrite.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
from functools import partial

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from agent import cloud
from agent.evidence_pipeline import EvidencePipeline
from agent.session import CallSession
from agent.fast_lane import WAIT_SYSTEM
from exp42_full_history import Recorder,evidence,run_turn
from windows_https_transport import native_client

OUT=ROOT/'experiments/results/exp44'
INTERRUPT_PREFIX=None


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def freeze(mode):
    cases=json.loads((ROOT/'experiments/results/exp35_plan.json').read_text(encoding='utf-8'))['cases']
    plan={'date':'2026-10-13','variant':'current v4 + full heard history + bounded waiting topic reply',
          'mode':mode,'frozen_cases':cases if mode=='text' else None,
          'waiting_prompt_sha256':hashlib.sha256(WAIT_SYSTEM.encode()).hexdigest(),
          'code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
             ['src/agent/session.py','src/agent/fast_lane.py','src/agent/consultation.py','src/agent/playback.py','src/agent/evidence_pipeline.py','src/agent/evidence_answer.py']},
          'transport':'certificate-checked Windows HTTPS for text models; Tencent ASR explicit direct WebSocket because automatic system proxy resets connection; original Qwen TTS',
          'scope':'Synthetic user input and injected echo; no physical microphone/telephone/customer holdout. Current single-turn voice results do not prove all long calls.'}
    path=OUT/(mode+'_plan.json')
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=plan:raise ValueError('Changed version: select a fresh --out directory')
    else:save(path,plan)
    return cases


def text_run(cases):
    def one(case):
        path=OUT/'text'/(case['id']+'.json')
        if path.exists():return {'id':case['id'],'existing':True}
        recorder=Recorder(inner=native_client())
        s=CallSession(EvidencePipeline(evidence(),client=recorder),voice=False,
                      profile=deepcopy(case['known']))
        row={'case':case,'opening':s.opening()}
        try:
            row['turn']=run_turn(s,case['question'])
            row['consultation']=deepcopy(s.consultation)
        except Exception as exc:row['error_type']=type(exc).__name__
        row['api_calls']=recorder.trace
        save(path,row)
        return {'id':case['id'],'error':row.get('error_type')}
    with ThreadPoolExecutor(4) as pool:
        for job in as_completed([pool.submit(one,c) for c in cases]):
            print(json.dumps(job.result(),ensure_ascii=False),flush=True)


def voice_run():
    import exp38_complex_conversation as voice
    import exp36_new_voice_acceptance as base
    from agent.fast_lane import CachedSpeech
    from agent.playback import PlaybackReceipts,TrackedDacPlayer,output_device
    import sounddevice as sd
    cloud.tencent_asr_stream=partial(cloud.tencent_asr_stream,proxy=None)
    record=ROOT/'experiments/recordings'/OUT.name
    record.mkdir(parents=True,exist_ok=True)
    voice.REC=record;base.REC=record/'user';base.REC.mkdir(exist_ok=True)
    recorder=Recorder(inner=native_client())
    voice.SESSION=CallSession(EvidencePipeline(evidence(),client=recorder))
    completed=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/'voice').glob('*_turn.json'))]
    if any(r.get('error_type') for r in completed):
        raise ValueError('Voice trial failed: preserve it and use a fresh output directory')
    if completed:
        latest=completed[-1]
        voice.SESSION.history=[tuple(h) for h in latest['history_after']]
        voice.SESSION.profile=deepcopy(latest['profile'])
        voice.SESSION.flags=deepcopy(latest['flags'])
        voice.SESSION.consultation=deepcopy(latest['consultation'])
        voice.SESSION.handoff_request=deepcopy(latest.get('handoff_request'))
        voice.SESSION.ended=latest['ended']
        save(OUT/'voice/resume.json',{'after_id':latest['id'],'restored_complete_heard_history':True,
            'reason':'Executor turn interrupted after a saved complete trial; completed samples are not rerun or overwritten.'})
    tts=cloud.RealtimeTTS(voice='Cherry',speech_rate=1.2)
    player=None
    try:
        cache=CachedSpeech(tts);cache.warm();voice.CACHE.update(cache.audio)
        if not completed:
            text=voice.SESSION.opening()
            receipt=PlaybackReceipts(voice.SESSION,voice.SESSION._playback_active)
            player=TrackedDacPlayer(receipt)
            token=receipt.enqueue(text);start=time.perf_counter()
            _,_,audio=tts.synth(text,on_audio=player.feed)
            player.sentence_end(token);end=player.drain()
            base.wav(record/'opening.wav',audio)
            save(OUT/'voice/00_opening.json',{'text':text,'audio_s':len(audio)/48000,
                 'playback_s':end-start,'device':dict(sd.query_devices(player.device))})
    finally:
        if player is not None:player.close()
        tts.close()
    for i,question in enumerate(voice.QUESTIONS,1):
        if i<=len(completed):continue
        if voice.SESSION.ended:break
        start_calls=len(recorder.trace)
        row=voice.voice_one({'id':f'T{i:02d}','question':question},1,evidence())
        row['api_calls']=recorder.trace[start_calls:]
        row['consultation']=deepcopy(voice.SESSION.consultation)
        save(OUT/'voice'/f'{i:02d}_turn.json',row)
        print(json.dumps({k:row.get(k) for k in ['id','first_response_dac_s','business_first_packet_s','actual_dac_playback_end_s','error_type','split_intended_turn']},ensure_ascii=False),flush=True)
        if row.get('error_type'):break
    save(OUT/'voice/final_session.json',{'history':voice.SESSION.history,'profile':voice.SESSION.profile,
                                      'consultation':voice.SESSION.consultation,'handoff_request':voice.SESSION.handoff_request})


def interrupt_run():
    import exp36_new_voice_acceptance as base
    import wave
    cloud.tencent_asr_stream=partial(cloud.tencent_asr_stream,proxy=None)
    record=ROOT/'experiments/recordings'/OUT.name
    record.mkdir(parents=True,exist_ok=True)
    base.REC=record/'user';base.REC.mkdir(exist_ok=True)
    voice_folder=(INTERRUPT_PREFIX/'voice') if INTERRUPT_PREFIX is not None else OUT/'voice'
    agents=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(voice_folder.glob('*_turn.json'))]
    agents=[a for a in agents if not a.get('error_type') and not a.get('split_intended_turn')]
    if not agents:raise ValueError('Need fresh voice replies')
    audio={}
    for a in agents:
        with wave.open(a['recording'],'rb') as w:audio[(a['id'],a['round'])]=w.readframes(w.getnframes())
    jobs=[]
    for kind,items in [('interrupt',base.INTERRUPTS),('backchannel',base.BACKCHANNELS),('echo',[None]*8)]:
        for i,text in enumerate(items):jobs.append((kind,text,.03,i,agents[i%len(agents)]))
    client=native_client()
    for job in jobs:
        key=f'g03_{job[0]}_{job[3]:02d}'
        path=OUT/'interrupt'/(key+'.json')
        if path.exists():continue
        if INTERRUPT_PREFIX is not None and (INTERRUPT_PREFIX/'interrupt'/(key+'.json')).exists():continue
        try:row=base.interrupt_one(job,audio,client)
        except Exception as exc:row={'key':key,'kind':job[0],'user':job[1],'error_type':type(exc).__name__}
        save(path,row)
        print(json.dumps({k:row.get(k) for k in ['key','kind','stopped','stop_latency_s','error_type']},ensure_ascii=False),flush=True)


def main():
    global OUT,INTERRUPT_PREFIX
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['text','voice','interrupt'],required=True)
    parser.add_argument('--out',default=str(OUT))
    parser.add_argument('--interrupt-prefix',help='Preserve already-recorded trials in this interrupted prefix; only execute absent trial IDs')
    args=parser.parse_args();OUT=Path(args.out);OUT.mkdir(parents=True,exist_ok=True)
    if args.interrupt_prefix:
        INTERRUPT_PREFIX=Path(args.interrupt_prefix)
        save(OUT/'interrupt_resume_source.json',{'prefix':str(INTERRUPT_PREFIX),
            'method':'Continue absent frozen trial IDs only, including earlier failures in final denominator. Never overwrite or select successful reruns.'})
    cases=freeze(args.mode)
    {'text':lambda:text_run(cases),'voice':voice_run,'interrupt':interrupt_run}[args.mode]()


if __name__=='__main__':main()
