"""New fast lane acceptance; real cloud and local DAC, synthetic users. No telephone validation."""
import asyncio, json, queue, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from exp36_new_voice_acceptance import *
from agent.fast_lane import CachedSpeech, FAST_PHRASES
from agent.playback import PlaybackReceipts, TrackedDacPlayer
from agent.session import _sentences
REC=ROOT/'experiments/recordings/exp37'
REC.mkdir(parents=True,exist_ok=True)
CACHE={}
def voice_one(case,round_,evidence):
    row={'id':case['id'],'round':round_,'question':case['question']}
    pcm=user_audio(case['question'],'Ethan' if round_==1 else 'Serena')
    wav(REC/f"input_{case['id']}_r{round_}.wav",pcm,16000)
    session=CallSession(EvidencePipeline(evidence),profile={'relation':'本人',**case['known']})
    session.flags.update(quoted=True,limits=True,disclosure=True)  # Consultation fixture; opening is not played here.
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
            profile=session.profile,timings=result[0].timings,
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
        wav(Path(row['recording']),b''.join(chunks))
    except Exception as exc:
        row['error_type']=type(exc).__name__;player.abort()
    finally:
        work.shutdown(wait=True);q.put(None);thread.join();tts.close();player.close()
    return row

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-only',action='store_true')
    parser.add_argument('--model-probe',action='store_true')
    parser.add_argument('--consultation-fix',action='store_true')
    parser.add_argument('--label')
    args=parser.parse_args()
    tts=cloud.RealtimeTTS(voice='Cherry',speech_rate=1.2)
    try:
        speech=CachedSpeech(tts);speech.warm();CACHE.update(speech.audio)
    finally:tts.close()
    evidence=[Evidence(**e) for e in json.loads((OUT/'exp33_organized_sources.json').read_text(encoding='utf-8'))['evidence']]
    plan=json.loads((OUT/'exp36_plan.json').read_text(encoding='utf-8'))
    cases=[c for c in plan['voice_cases'] if c['id'] in ('C02','H07','H15')]
    cases.append({'id':'TRANSFER','question':'请帮我转人工客服','known':{}})
    if args.model_only:
        cases=[{'id':'MODEL_ACK','question':'想了解癌症相关保障','known':{'age':30,'has_social':True}}]
    label='exp37_model' if args.model_only else 'exp37'
    if args.model_probe:
        cases=[{'id':'MODEL_ACK2','question':'癌症能报不','known':{'age':30,'has_social':True}}]
        label='exp37_model2'
    if args.consultation_fix:
        cases=[{'id':'CONSULTATION_FIX','question':'想了解癌症相关保障','known':{'age':30,'has_social':True}}]
        label='exp37_consultation_fix'
    if args.label:
        label=args.label
    folder=OUT/(label+'_fast_lane_records');folder.mkdir(exist_ok=True)
    save(OUT/(label+'_plan.json'),{'cases':cases,'code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/agent/session.py','src/agent/fast_lane.py','src/agent/evidence_pipeline.py']},'method':'Cached neutral acknowledgment plus unchanged DS evidence answer; actual ASR/TTS and local DAC. Transfer requests recorded only. Business first packet separately measured, not business DAC.'})
    for case in cases:
        row=voice_one(case,1,evidence)
        save(folder/(case['id']+'.json'),row)
        print(json.dumps({k:row.get(k) for k in ['id','first_response_dac_s','business_first_packet_s','actual_dac_playback_end_s','split_intended_turn','error_type']},ensure_ascii=False),flush=True)
