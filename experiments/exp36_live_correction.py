"""Two real playback interruptions followed by ASR corrections on the same session."""
import asyncio
import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from agent import cloud,small_model
from agent.evidence_answer import Evidence
from agent.evidence_pipeline import EvidencePipeline
from agent.session import CallSession
from agent.dialogue import BargeInController
from exp30_fewshot import save
from exp36_new_voice_acceptance import DacPlayer,user_audio,REC,wav
from exp5_audio import mix

OUT=ROOT/'experiments/results'
CASES=[('social','等一下我没有社保我30岁首次年缴多少钱',679),
    ('age','不对我不是30岁我是40岁有社保首次年缴多少钱',517)]

def main():
    plan={'cases':[{'id':i,'correction':t,'expected_price_not_sent':p} for i,t,p in CASES],
        'initial':'我30岁有社保第一次买基础版年缴多少钱','echo_gain':.03,
        'method':'Fresh base answer on real session; real TTS playback and live mixed-ASR interruption; same session receives confirmed correction and generates fresh reply. Session history is inspected for falsely retained unheard full text. No history repair or model judge.'}
    path=OUT/'exp36_live_correction_plan.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==plan
    else:save(path,plan)
    evidence=[Evidence(**e) for e in json.loads((OUT/'exp33_organized_sources.json').read_text(encoding='utf-8'))['evidence']]
    folder=OUT/'exp36_live_correction_records';folder.mkdir(exist_ok=True)
    for i,question,price in CASES:
        path=folder/(i+'.json')
        if path.exists():continue
        row={'id':i,'correction':question,'expected_price_not_sent':price}
        session=CallSession(EvidencePipeline(evidence),profile={'relation':'本人','age':30,'has_social':True})
        session.opening();session.flags.update(quoted=True,limits=True,disclosure=True)
        tts=cloud.RealtimeTTS(voice='Cherry',speech_rate=1.2);player=DacPlayer();pool=ThreadPoolExecutor(1)
        try:
            base=session.turn(plan['initial'])
            _,_,audio=tts.synth(base.text)
            wav(REC/f'correction_base_{i}.wav',audio)
            upcm=user_audio(question,'Ethan')
            mic,onset,agent_end=mix(cloud.resample_24k_to_16k(audio),upcm,1.2,.03)
            origin=[None];stopped=[None];decisions=[];futures=[]
            ctrl=BargeInController(lambda t:small_model.classify_barge_in(t,client=session.pipe.client))
            ctrl.note_spoken(0,base.text)
            def decide(ev):
                if stopped[0] is not None:return
                now=time.perf_counter()-origin[0]
                d=ctrl.on_partial(now,ev.text,now<len(audio)/48000)
                if d is None:return
                decisions.append({'action':d.action,'asr':ev.text,'time_s':time.perf_counter()-origin[0],'reason':d.reason})
                if d.action=='stop':stopped[0]=player.abort()
            def event(ev):
                if ev.text:futures.append(pool.submit(decide,ev))
            def frame(data,at):
                if origin[0] is None:
                    origin[0]=time.perf_counter()-(at-len(data)/32000);player.feed(audio)
            events,_=asyncio.run(cloud.tencent_asr_stream(mic,vad_silence_ms=500,
                trailing_silence_ms=0,on_event=event,on_audio=frame))
            for f in futures:f.result()
            finals=[e.text for e in events if e.slice_type==2]
            # Preserve exact ASR; select the first confirmed correction, never insert gold text.
            corrected=next((t for t in finals if ('没有社保' in t if i=='social' else '40' in t or '四十' in t)),None)
            stopped_early=bool(stopped[0] and stopped[0]['dropped_frames']>0)
            falsely_full_history=stopped_early and session.history[-1][1]==base.text
            player.close();player=DacPlayer()
            spoken=[];q=queue.Queue();audio_out=[];tts_errors=[]
            def speak():
                while True:
                    text=q.get()
                    if text is None:return
                    try:audio_out.append(tts.synth(text,on_audio=player.feed)[2])
                    except Exception as exc:tts_errors.append(type(exc).__name__)
            worker=threading.Thread(target=speak);worker.start()
            def emit(text):spoken.append(text);q.put(text)
            if corrected is not None:
                reply=session.turn(corrected,on_sentence=emit)
                q.put(None);worker.join();player.drain()
                wav(REC/f'correction_reply_{i}.wav',b''.join(audio_out))
                row.update(reply_text=reply.text,reply_timings=reply.timings,reply_events=reply.events)
            else:q.put(None);worker.join()
            row.update(base_text=base.text,base_timings=base.timings,asr_events=[vars(e) for e in events],
                corrected_asr_text=corrected,profile=session.profile,history=session.history,
                decisions=decisions,stopped_before_natural_end=stopped_early,
                stop_latency_s=stopped[0]['last_scheduled_dac_wall']-origin[0]-onset if stopped[0] else None,
                cancellation=stopped[0],unheard_full_answer_retained_in_history=falsely_full_history,
                spoken_chunks=spoken,tts_errors=tts_errors)
        except Exception as exc:row['error_type']=type(exc).__name__
        finally:pool.shutdown(wait=True);player.abort();player.close();tts.close()
        save(path,row);print(json.dumps({k:row.get(k) for k in ['id','corrected_asr_text','stop_latency_s','reply_text','profile','unheard_full_answer_retained_in_history','error_type']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
