"""Recompute text latency from saved calls, without new cloud/audio requests."""
import json
import math
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent/'results'

def main():
    plan = json.loads((OUT/'exp35_plan.json').read_text(encoding='utf-8'))
    control = [json.loads((OUT/'exp34_records'/('control_'+c['id']+'.json')).read_text(encoding='utf-8')) for c in plan['cases']]
    candidate = [json.loads((OUT/'exp35_records'/(c['id']+'.json')).read_text(encoding='utf-8')) for c in plan['cases']]
    def summary(values):
        values = sorted(values)
        return {'n':len(values),'median':statistics.median(values),
            'p90':values[math.ceil(.9*len(values))-1],
            'p95':values[math.ceil(.95*len(values))-1],'max':max(values)}
    result = {
        'control_full_text_s':summary([r['total_s'] for r in control]),
        'candidate_full_text_s':summary([r['total_s'] for r in candidate]),
        'paired_delta_s':summary([b['total_s']-a['total_s'] for a,b in zip(control,candidate)]),
        'candidate_above_3s':sum(r['total_s']>3 for r in candidate),
        'control_chars':summary([len(r['text']) for r in control]),
        'candidate_chars':summary([len(r['text']) for r in candidate]),
        'candidate_local_after_call_s':summary([r['total_s']-sum(c['seconds'] for c in r['calls']) for r in candidate]),
        'actions':{action:summary([r['total_s'] for r in candidate if r['outcome']['action']==action]) for action in ['answer','clarify','handoff']},
        'candidate_one_call_every_row':all(len(r['calls'])==1 for r in candidate),
        'method':'Existing exp34 control and exp35 candidate; sequential same-case comparison, not new calls. Complete text, not first token, STT/TTS or customer first audio. Local overhead includes rendering, provenance and decision validation, not just handoff.'}
    (OUT/'exp35_latency_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
