"""One persistent full consultation, real ASR/DS/TTS/DAC, synthetic user input."""
import asyncio, json, queue, threading, time, hashlib
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from exp36_new_voice_acceptance import *
from agent.fast_lane import CachedSpeech, FAST_PHRASES
from agent.playback import PlaybackReceipts, TrackedDacPlayer
from agent.session import _sentences
REC=ROOT/'experiments/recordings/exp38'
REC.mkdir(parents=True,exist_ok=True)
CACHE={}
SESSION=None
QUESTIONS=[
'我是给自己了解的我30岁有社保首次投保必选计划一年多少钱',
'如果以后得了癌症住院治疗靶向药和质子重离子这三项分别怎么赔各有没有免赔额医院有要求吗',
'假设符合一般住院责任医院总费用两万元医保先报了八千元剩余一万二千元都是合同范围内费用那么这个保险还能报多少钱',
'我更正一下我实际是40岁没有社保首次投保必选计划年缴和月缴分别多少钱',
'按月交十二个月比年缴多多少钱另外生病住院和意外住院都要等三十天吗',
'如果投保前就有高血压以后发生与高血压无关的交通意外住院是不是一定都不赔请说明适用条件',
'请确认原位癌是否属于合同约定的一百种重大疾病资料缺少定义的话就转人工不要猜'
]
def voice_one(case,round_,evidence):
    row={'id':case['id'],'round':round_,'question':case['question'],'history_before':list(SESSION.history),'profile_before':dict(SESSION.profile)}
    pcm=user_audio(case['question'],'Ethan' if round_==1 else 'Serena')
    wav(REC/f"input_{case['id']}_r{round_}.wav",pcm,16000)
    session=SESSION
    receipts=PlaybackReceipts(session,None)
    player=TrackedDacPlayer(receipts);tts=cloud.RealtimeTTS(voice='Cherry',speech_rate=1.2)
    speech=CachedSpeech(tts);speech.audio.update(CACHE)
    q=queue.Queue();chunks=[];errors=[];origin=[None];result=[None];task=[None]
    final=[None];text_done=[None];sentences=[]
    def emit(text):
        if receipts.playback_id is None:receipts.playback_id=session._playback_active
        for fragment in _sentences(text):
            token=receipts.enqueue(fragment)
            if token is not None:q.put((fragment,time.perf_counter(),token))
    def worker():
        while True:
            item=q.get()
            if item is None:return
            text,emitted,token=item;t0=time.perf_counter()
            player.sentence_start(token)
            received=[]
            def packet(audio):
                received.append(time.perf_counter());player.feed(audio)
            try:
                first,done,audio=speech.synth(text,on_audio=packet)
                player.sentence_end(token)
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
            profile=dict(session.profile),timings=result[0].timings,
            history_after=list(session.history),handoff_request=session.handoff_request,ended=session.ended,
            text_ready_s=text_done[0]-user_end,
            asr_endpoint_s=origin[0]+final[0].t-user_end,
            first_response_pcm_s=min(s['first_packet_wall'] for s in sentences if s['first_packet_wall'])-user_end,
            first_response_dac_s=player.first_dac-user_end if player.first_dac else None,
            actual_dac_playback_end_s=end-user_end,
            business_first_packet_s=min((s['first_packet_wall']-user_end for s in sentences if s['text'] not in FAST_PHRASES and s['first_packet_wall']),default=None),
            audio_duration_s=sum(s['duration_s'] for s in sentences),
            sentences=[{k:(v-user_end if k.endswith('_wall') and v is not None else v) for k,v in s.items()} for s in sentences],
            playback_statuses=player.statuses,output_latency_s=player.stream.latency,
            recording=str(REC/f"answer_{case['id']}_r{round_}.wav"))
        bridge=result[0].timings.get('waiting_reply',{})
        bridge_text=bridge.get('text') if bridge.get('emitted') else None
        row['legacy_first_nonfixed_packet_s']=row['business_first_packet_s']
        row['business_first_packet_s']=min((s['first_packet_wall']-user_end for s in sentences
            if s['text'] not in FAST_PHRASES and s['text']!=bridge_text and s['first_packet_wall']),default=None)
        row['waiting_first_packet_s']=min((s['first_packet_wall']-user_end for s in sentences
            if s['text']==bridge_text and s['first_packet_wall']),default=None)
        row['business_first_dac_s']=min((f['dac_first_audible']-user_end for f in receipts.fragments
            if f['text'] not in FAST_PHRASES and f['text']!=bridge_text and f['dac_first_audible']),default=None)
        row['receipt_fragments']=[{**f,'dac_end_s':f['dac_end']-user_end if f['dac_end'] else None,
            'dac_first_audible_s':f['dac_first_audible']-user_end if f['dac_first_audible'] else None} for f in receipts.fragments]
        wav(Path(row['recording']),b''.join(chunks))
    except Exception as exc:
        row['error_type']=type(exc).__name__
        if isinstance(exc,ValueError) and str(exc).startswith(('Playback receipt','Full playback','Output still')):
            row['safe_error_message']=str(exc)
        import traceback
        row['error_stack']=[{'file':Path(f.filename).name,'function':f.name,'line':f.lineno}
                            for f in traceback.extract_tb(exc.__traceback__)]
        row['failed_playback_record']=deepcopy(session._playback_records.get(session._playback_active,{}))
        row['failed_session_history']=deepcopy(session.history)
        player.abort()
    finally:
        work.shutdown(wait=True);q.put(None);thread.join();tts.close();player.close()
    return row

if __name__=='__main__':
    evidence=[Evidence(**e) for e in json.loads((OUT/'exp33_organized_sources.json').read_text(encoding='utf-8'))['evidence']]
    SESSION=CallSession(EvidencePipeline(evidence))
    folder=OUT/'exp38_conversation_records';folder.mkdir(exist_ok=True)
    plan={'questions':QUESTIONS,'model':cloud.SLOW_MODEL,'date':'2026-10-12','profile_start':{},'flags_manually_set':False,'method':'One persistent CallSession from audible opening to natural end/handoff. Synthetic user audio, real cloud ASR/flow/DS/TTS and actual WASAPI playback. No telephone, human connection or concurrent user interruption. First DAC and business first TTS packet are distinct metrics. Stops immediately if session ends; no prompted recovery or selected reruns.','code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/agent/session.py','src/agent/fast_lane.py','src/agent/evidence_pipeline.py','src/agent/evidence_answer.py']},'evidence_sha256':hashlib.sha256((OUT/'exp33_organized_sources.json').read_bytes()).hexdigest()}
    save(OUT/'exp38_plan.json',plan)
    tts=cloud.RealtimeTTS(voice='Cherry',speech_rate=1.2)
    try:
        speech=CachedSpeech(tts);speech.warm();CACHE.update(speech.audio)
        opening=SESSION.opening();opening_receipts=PlaybackReceipts(SESSION,SESSION._playback_active)
        player=TrackedDacPlayer(opening_receipts);start=time.perf_counter()
        try:
            token=opening_receipts.enqueue(opening)
            _,_,audio=tts.synth(opening,on_audio=player.feed)
            player.sentence_end(token)
            end=player.drain();wav(REC/'opening.wav',audio)
            save(folder/'00_opening.json',{'speaker':'agent','text':opening,'first_dac_s':player.first_dac-start,'playback_done_s':end-start,'audio_s':len(audio)/48000,'recording':str(REC/'opening.wav')})
        finally:player.close()
    finally:tts.close()
    rows=[]
    for i,question in enumerate(QUESTIONS,1):
        if SESSION.ended:break
        row=voice_one({'id':f'T{i:02d}','question':question,'known':{}},1,evidence)
        rows.append(row);save(folder/f'{i:02d}_turn.json',row)
        print(json.dumps({k:row.get(k) for k in ['id','first_response_dac_s','business_first_packet_s','actual_dac_playback_end_s','split_intended_turn','error_type','ended']},ensure_ascii=False),flush=True)
        if row.get('error_type'):break
    save(OUT/'exp38_conversation.json',{'method':plan['method'],'turns':rows,'history':SESSION.history,'final_profile':SESSION.profile,'flags':SESSION.flags,'ended':SESSION.ended,'handoff_request':SESSION.handoff_request,'unexecuted_questions':QUESTIONS[len(rows):]})
