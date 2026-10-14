"""Small live planner regression against the recorded exp44 history failure.

Checks task scope and IDs, not insurance answer correctness or voice latency.
Never overwrites an existing result. Requires the configured Qwen service.
"""
import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agent import cloud
from agent.consultation import add_questions
from agent.pipeline import understand_split


def cases():
    recorded=json.loads((ROOT/'experiments/results/exp44_voice_current/voice/06_turn.json').read_text(encoding='utf-8'))
    state=deepcopy(next(c['known']['_consultation'] for c in recorded['api_calls'] if '_consultation' in c.get('known',{})))
    # Freeze the valid ledger immediately before the spurious historical seeds.
    state['tasks']=[t for t in state['tasks'] if not t['id'].startswith('c7-u4-')]
    known={**recorded['profile_before'],'_consultation':state,
           '_history':'\n'.join(f'{speaker}：{text}' for speaker,text in recorded['history_before'])}
    repeat=next(t for t in state['tasks'] if t['id']=='c7-price-2')
    return known,[
        {'id':'recorded-T06','text':recorded['question'],'required':['高血压','意外'],
         'forbidden':['靶向','质子','重离子','月缴','年缴']},
        {'id':'conditions-only','text':'我40岁，没有社保。','empty':True},
        {'id':'backchannel','text':'嗯，我知道了。','empty':True},
        {'id':'repeat-monthly','text':'刚才月缴多少钱，再说一次。','required':['月'],
         'reuse':repeat['id'],'forbidden':['靶向','质子','重离子']},
        {'id':'new-compound','text':'药省保每月最多报多少，ICU津贴每天多少钱？',
         'required':['药省保','ICU'],'min_count':2,'forbidden':['靶向','质子','重离子']},
    ]


def run(out,native_https=False):
    out=Path(out)
    if out.exists():
        raise FileExistsError(f'Result already exists: {out}')
    known,tests=cases()
    out.parent.mkdir(parents=True,exist_ok=True)
    # Claim the destination before making calls; partial results survive failures.
    out.touch(exist_ok=False)
    if native_https:
        from windows_https_transport import WindowsHttpsTransport
        import httpx
        client=cloud.qwen_client().with_options(http_client=httpx.Client(transport=WindowsHttpsTransport()))
    else:
        client=cloud.qwen_client()
    records=[]
    for case in tests:
        calls=[]
        original_chat=cloud.chat
        def observe(messages,**kwargs):
            is_planner='obligations' in messages[0]['content']
            try:
                raw=original_chat(messages,**kwargs)
                calls.append({'planner':is_planner,'output':json.loads(raw)})
                return raw
            except Exception as exc:
                calls.append({'planner':is_planner,'error':type(exc).__name__})
                raise
        begin=time.perf_counter()
        with patch('agent.pipeline.cloud.chat',side_effect=observe):
            understanding=understand_split(case['text'],deepcopy(known),client)
        plan=understanding.get('obligation_plan',[])
        joined=' '.join(p['question'] for p in plan)
        faults=[]
        if not any(c['planner'] and 'output' in c for c in calls):faults.append('planner_service_failed')
        if case.get('empty') and plan:faults.append('background_became_question')
        for keyword in case.get('required',[]):
            if keyword.lower() not in joined.lower():faults.append('missing:'+keyword)
        for keyword in case.get('forbidden',[]):
            if keyword in joined:faults.append('historical_topic:'+keyword)
        if len(plan)<case.get('min_count',0):faults.append('compound_question_dropped')
        if case.get('reuse') and not any(p['task_id']==case['reuse'] for p in plan):faults.append('repeat_id_not_reused')
        state=deepcopy(known['_consultation'])
        add_questions(state,plan,case['text'])
        record={'case':case,'plan':plan,'faults':faults,'calls':calls,
                'seconds':round(time.perf_counter()-begin,3),'tasks_after':state['tasks']}
        records.append(record)
        result={'model':cloud.FAST_MODEL,'scope':'planner_only','native_https':native_https,
                'input':known,'records':records,'passed':sum(not r['faults'] for r in records),'total':len(records)}
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'case':case['id'],'plan':plan,'faults':faults},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='experiments/results/exp45_question_planning.json')
    parser.add_argument('--native-https',action='store_true')
    args=parser.parse_args()
    result=run(args.out,args.native_https)
    print(f"Passed {result['passed']}/{result['total']}")
    sys.exit(0 if result['passed']==result['total'] else 1)
