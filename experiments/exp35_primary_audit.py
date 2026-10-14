"""Aggregate primary-agent reading and pre-frozen routing policy, not model grading."""
import hashlib
import json
import math
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent/'results'

def main():
    plan = json.loads((OUT/'exp35_plan.json').read_text(encoding='utf-8'))
    decisions = json.loads((OUT/'exp35_manual_decisions.json').read_text(encoding='utf-8'))
    policies = {p['id']:p for p in plan['policies_not_sent_to_model']}
    rows = []
    for variant, folder, prefix in [('control','exp34_records','control_'),('post_handoff_v2','exp35_records','')]:
        assert set(decisions[variant]) == set(policies)
        for case in plan['cases']:
            path = OUT/folder/(prefix+case['id']+'.json')
            raw = path.read_bytes()
            r = json.loads(raw)
            content, trace, reason = decisions[variant][case['id']]
            action = 'answer' if variant=='control' else r.get('outcome',{}).get('action')
            valid = not r.get('error_type') and (variant=='control' or r.get('outcome_validation',{}).get('valid',False))
            proper = valid and action in policies[case['id']]['allowed_actions']
            # Subsequent explicit user policy accepts a safe handoff even when
            # the question was answerable. Preserve the frozen original score.
            user_accepted = valid and (proper or action=='handoff')
            rows.append({'id':case['id'],'variant':variant,'group':case['group'],
                'content_correct':content,'traceable':trace,'joint_correct':content and trace,
                'decision_valid':valid,'action':action,'decision_correct':proper,
                'automatically_resolved':proper and action=='answer' and content and trace,
                'handled_correctly':proper and content and trace,
                'handled_under_user_policy':user_accepted and content and trace,
                'required_external':policies[case['id']]['required_external'],
                'reason':reason,'answer':r.get('text'),'record_sha256':hashlib.sha256(raw).hexdigest(),
                'total_s':r.get('total_s')})
    def metrics(rs):
        def score(key, subset=None):
            selected = rs if subset is None else subset
            count = sum(r[key] for r in selected)
            return {'passed':count,'n':len(selected),'percent':round(100*count/len(selected),2) if selected else None}
        answered = [r for r in rs if r['action']=='answer' and r['decision_valid']]
        handed = [r for r in rs if r['action']=='handoff' and r['decision_valid']]
        required = [r for r in rs if r['required_external']]
        times = sorted(r['total_s'] for r in rs if r['total_s'] is not None)
        return {'n':len(rs),**{key:score(key) for key in ['content_correct','traceable','joint_correct','decision_correct','automatically_resolved','handled_correctly','handled_under_user_policy']},
            'automatic_answer_content_accuracy':score('content_correct',answered),
            'automatic_answer_joint_accuracy':score('joint_correct',answered),
            'answers_issued':len(answered),'handoffs_issued':len(handed),
            'appropriate_handoff_decisions':score('decision_correct',handed),
            'appropriate_traceable_safe_handoffs':score('handled_correctly',handed),
            'accepted_traceable_safe_handoffs_user_policy':score('handled_under_user_policy',handed),
            'handoff_rate_percent':round(100*len(handed)/len(rs),2),
            'required_handoffs_requested':{'requested':sum(r['action']=='handoff' and r['decision_valid'] for r in required),'n':len(required)},
            'missed_required_handoffs':[r['id'] for r in required if r['action']!='handoff' or not r['decision_valid']],
            'unnecessary_handoffs':[r['id'] for r in handed if not r['decision_correct']],
            'clarifications':[r['id'] for r in rs if r['action']=='clarify'],
            'content_failures':[r['id'] for r in rs if not r['content_correct']],
            'trace_failures':[r['id'] for r in rs if not r['traceable']],
            'decision_failures':[r['id'] for r in rs if not r['decision_correct']],
            'full_text_latency_s':{'median':statistics.median(times),'p95':times[math.ceil(.95*len(times))-1]}}
    summary={v:{'original127':metrics([r for r in rows if r['variant']==v and r['group']!='new_handoff16']),
        'new16':metrics([r for r in rows if r['variant']==v and r['group']=='new_handoff16']),
        'all143':metrics([r for r in rows if r['variant']==v])} for v in decisions}
    pilot = json.loads((OUT/'exp34_post_handoff.json').read_text(encoding='utf-8'))['rows']
    result={'method':'Primary agent read all 286 comparison answers and checked source support. No independent second reviewer. Each claim uses its own attached citations; whole-response content and decision evaluated separately.',
        'design':plan['design'],'summary':summary,
        'user_policy_revision':{'date':'2026-10-11','instruction':'可以答的转人工了也算对',
            'definition':'A valid requested handoff is accepted even if answerable; spoken facts must remain correct and traceable. Automatic resolution and answer accuracy are unchanged. Original frozen decision and handling metrics retained for comparison.'},
        'pilot_v1_schema_failures':[r['id'] for r in pilot if r['variant']=='post_handoff' and r.get('error_type')],
        'pilot_note':'11 ValueErrors counted as failures. Old runner retained exception type only, so precise invalid fields and discarded raw outputs cannot be reconstructed. No v1 accuracy claim.',
        'limits':['Fixed development and synthetic diagnostic questions, not customer holdout.',
            'Control reused from exp34, candidate from later exp35: sequential comparison.',
            'Frozen policy can require transfer despite a factually safe bounded response; report both metrics.',
            'Correct handling includes appropriate clarification or requested handoff; it is not automatic answer accuracy or resolved customer claim.',
            'Transfer request saved only; no phone provider connection tested. Opt-in adapter, production default unchanged.',
            'Timing is full text completion, not speech first response. STT/TTS not invoked.',
            'E09/E19 retain personal residual expense interpretation; ambiguity remains.'],
        'rows':rows}
    (OUT/'exp35_primary_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
