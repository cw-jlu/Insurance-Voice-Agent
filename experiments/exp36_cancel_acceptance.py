"""Real DS in-flight speculative cancellation; no simulated model or auto judge."""
import json
import sys
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from agent.evidence_answer import Evidence, EvidenceAnswerer
from agent.evidence_pipeline import EvidencePipeline
from agent.session import CallSession
from agent.speculative import SpeculativeTurn
from exp30_fewshot import save

OUT=ROOT/'experiments/results'
CASES=[
    ('social_year','我30岁有社保第一次买基础版年缴多少钱','不对我没有社保我30岁首次年缴多少钱',679),
    ('age_40','我30岁有社保第一次买基础版年缴多少钱','我不是30岁我是40岁有社保首次年缴多少钱',517),
    ('pay_month','我30岁有社保第一次买基础版年缴多少钱','我想按月交30岁有社保首次每月多少钱',31),
    ('age_69','我30岁有社保第一次买基础版年缴多少钱','我今年69岁有社保首次买基础版年缴多少钱',3316),
]

def main():
    plan={'cases':[{'id':i,'initial':a,'corrected':b,'expected_price_not_sent':price} for i,a,b,price in CASES],
        'design':'Start real unconfirmed preparation, wait for DS request to start, invalidate it, confirm correction, inspect emitted text and committed state. This is generation cancellation, not microphone or playback stopping.'}
    path=OUT/'exp36_cancel_plan.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==plan
    else:save(path,plan)
    evidence=[Evidence(**e) for e in json.loads((OUT/'exp33_organized_sources.json').read_text(encoding='utf-8'))['evidence']]
    folder=OUT/'exp36_cancel_records';folder.mkdir(exist_ok=True)
    for i,initial,corrected,price in CASES:
        path=folder/(i+'.json')
        if path.exists():continue
        engine=EvidenceAnswerer();original=engine.call;started=threading.Event();calls=[];lock=threading.Lock()
        def call(system,user,model):
            is_initial=user.startswith('原问题：'+initial+'\n')
            at=time.perf_counter()
            if is_initial:started.set()
            result=original(system,user,model)
            with lock:calls.append({'initial':is_initial,'start_wall':at,'finish_wall':time.perf_counter(),'result':result})
            return result
        engine.call=call
        pipe=EvidencePipeline(evidence,answerer=engine)
        session=CallSession(pipe,profile={'relation':'本人','age':30,'has_social':True})
        session.opening();session.flags.update(quoted=True,limits=True,disclosure=True)
        manager=SpeculativeTurn(session,streaming=True)
        row={'id':i,'initial':initial,'corrected':corrected,'expected_price_not_sent':price};spoken=[]
        try:
            manager.partial(initial)
            candidate=manager.candidate
            if not started.wait(30):raise TimeoutError('Initial DS call did not start')
            cancelled=time.perf_counter();manager.invalidate(corrected)
            result=manager.confirm(corrected,spoken.append)
            manager.close()
            initial_calls=[c for c in calls if c['initial']]
            row.update(cancel_wall=cancelled,in_flight_at_cancel=bool(initial_calls and initial_calls[0]['start_wall']<=cancelled<initial_calls[0]['finish_wall']),
                calls=calls,stats=manager.stats,spoken_chunks=spoken,text=result.text,
                profile=manager.session.profile,history=manager.session.history,timings=result.timings,
                cancelled_preparation=candidate.cancelled.is_set(),
                emitted_text_matches_committed=''.join(spoken)==result.text,
                initial_text_released=any(c['result']['output'].get('segments') and ''.join(s['text'] for s in c['result']['output']['segments']) in ''.join(spoken) for c in initial_calls))
        except Exception as exc:
            row['error_type']=type(exc).__name__
            manager.close()
        save(path,row);print(json.dumps({k:row.get(k) for k in ['id','in_flight_at_cancel','initial_text_released','text','profile','error_type']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
