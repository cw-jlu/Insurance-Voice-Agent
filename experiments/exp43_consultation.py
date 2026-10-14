"""Real-model consultation validation, with replies triggered by audible questions.

No hidden facts/oracles are sent to the agent; records are never overwritten.
Uses certificate-checked Windows HTTPS only for the local test transport.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments')]
import exp42_full_history as prior
from agent.evidence_pipeline import EvidencePipeline
from agent.session import CallSession
from windows_https_transport import native_client

OUT = ROOT/'experiments/results/exp43_records'
FIELDS = {**prior.FIELDS,
    'age':r'多大|几岁|多少岁|年龄|周岁|岁数',
    'qualified_family':r'同一投保人|同一.*订单|已有保单|适用.*系列|累计.*成员|符合.*家庭',
    'remaining_deductible':r'免赔|(?:剩|使用|抵扣).{0,15}免赔',
}


def run_case(case):
    path = OUT/(case['id']+'.json')
    if path.exists():
        return {'id':case['id'],'existing':True}
    recorder = prior.Recorder(inner=native_client())
    session = CallSession(EvidencePipeline(prior.evidence(),client=recorder),voice=False,
                          profile=deepcopy(case['initial_profile']),
                          history=[tuple(h) for h in case['initial_history']])
    row = {'id':case['id'],'type':case['type'],'split':case['split'],
           'opening':session.opening(),'turns':[],'operator_actions':[]}
    question, used = case['initial_user'], set()
    try:
        for _ in range(12):
            turn = prior.run_turn(session,question)
            turn['consultation'] = deepcopy(session.consultation)
            row['turns'].append(turn)
            if session.ended:
                row['stop_reason'] = 'session_ended'
                break
            detail = turn['timings'].get('answer_detail',{}).get('evidence_result',{})
            outcome = detail.get('effective_outcome',detail.get('outcome',{}))
            allow_correction = outcome.get('action')=='answer' and session.consultation.get('status')=='complete'
            # Metadata is not speech: hidden facts are disclosed only for an
            # audible request, or a predeclared correction after an actual answer.
            questions = [s for s in re.split(r'(?<=[。！？!?])',turn['text'])
                         if '?' in s or '？' in s or re.search(r'请(?:告诉|提供|补充|确认)',s)]
            search = '\n'.join(questions)
            selected = []
            for i,rule in enumerate(case['user_reply_rules']):
                if i in used:continue
                # A later person's conditions cannot be disclosed before the
                # scripted correction that introduces that person.
                if any(j not in used and earlier['mode']=='after_answer'
                       for j,earlier in enumerate(case['user_reply_rules'][:i])):
                    continue
                if rule['mode']=='after_answer':
                    if allow_correction:
                        selected = [i]
                        break
                elif any(re.search(FIELDS.get(f,r'(?!)'),search) for f in rule['requested_fields']):
                    selected.append(i)
            if not selected:
                row['stop_reason'] = 'no_relevant_audible_question_or_all_conditions_disclosed'
                break
            used.update(selected)
            question = ''.join(case['user_reply_rules'][i]['utterance'] for i in selected)
            row['operator_actions'].append({'reply_indices':selected,'utterance':question,
                                           'triggering_outcome':outcome,'triggering_text':turn['text']})
        else:row['stop_reason']='diagnostic_round_cap'
    except Exception as exc:
        row['error_type'] = type(exc).__name__
    row['undisclosed_reply_indices'] = [i for i in range(len(case['user_reply_rules'])) if i not in used]
    row['api_calls'] = recorder.trace
    path.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'id':case['id'],'turns':len(row['turns']),'stop':row.get('stop_reason'),'error':row.get('error_type')}


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids',nargs='*')
    parser.add_argument('--out',default=str(OUT))
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    OUT=Path(args.out)
    OUT.mkdir(parents=True,exist_ok=True)
    data=json.loads((ROOT/'experiments/data/complex_dialogue_v1.json').read_text(encoding='utf-8'))
    cases=[c for c in data['cases'] if not args.ids or c['id'] in args.ids]
    plan={'date':'2026-10-13','models':{'answer':prior.cloud.SLOW_MODEL,'flow':prior.cloud.FAST_MODEL},
          'variant':'v4 persistent consultation; full heard history; overflow-only compression',
          'case_ids':[c['id'] for c in cases],'no_audio':True,
          'operator_rules_not_sent_to_agent':True,'operator':'audible questions only; raw responses retained',
          'transport':'Windows certificate-checked HTTPS test-only; no production networking change',
          'code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
             ['src/agent/consultation.py','src/agent/session.py','src/agent/evidence_pipeline.py','src/agent/evidence_answer.py']}}
    plan_path=OUT/'plan.json'
    if plan_path.exists():
        old=json.loads(plan_path.read_text(encoding='utf-8'))
        if old['code_sha256']!=plan['code_sha256'] or not set(plan['case_ids']).issubset(old['case_ids']):
            raise ValueError('Use a fresh --out directory for a changed version/case set; original records are immutable')
    else:
        plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run_case,cases):
            print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
