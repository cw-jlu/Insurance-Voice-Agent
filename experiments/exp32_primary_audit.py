"""Aggregate primary-agent recorded judgments, without automated answer grading."""
import hashlib
import json
import math
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'results'

def main():
    raw = (OUT/'exp32_fewshot_v2.json').read_bytes()
    data = json.loads(raw)
    decisions = json.loads((OUT/'exp32_manual_decisions.json').read_text(encoding='utf-8'))
    prior = json.loads((OUT/'exp31_primary_audit.json').read_text(encoding='utf-8'))
    assert len(data['rows']) == len(decisions) == 135
    reviewed = []
    for row in data['rows']:
        c,t,reason = decisions[row['id']]
        reviewed.append({'id':row['id'],'group':row['group'],'content_correct':c,
            'traceable':t,'joint_correct':c and t,'reason':reason,'answer':row['text'],
            'record_sha256':hashlib.sha256((OUT/'exp32_records'/f"{row['id']}.json").read_bytes()).hexdigest()})
    def metrics(rows):
        n=len(rows)
        return {'n':n,**{k:{'passed':sum(r[k] for r in rows),'percent':round(100*sum(r[k] for r in rows)/n,2)} for k in ['content_correct','traceable','joint_correct']},
            'content_failures':[r['id'] for r in rows if not r['content_correct']],
            'trace_failures':[r['id'] for r in rows if not r['traceable']]}
    original=[r for r in data['rows'] if r['group']!='transfer8']
    times=sorted(r['total_s'] for r in original)
    summary={'original127':metrics([r for r in reviewed if r['group']!='transfer8']),
        'groups':{g:metrics([r for r in reviewed if r['group']==g]) for g in ['development71','prior_boundary16','new40','transfer8']},
        'latency_original127_s':{'median':statistics.median(times),'p95_nearest_rank':times[math.ceil(.95*len(times))-1],'max':max(times)},
        'median_chars_original127':statistics.median(len(r['text']) for r in original),
        'mechanical_provenance_pass_original127':sum(r['provenance']['valid'] for r in original),
        'call_errors':[r['id'] for r in data['rows'] if r.get('error_type')],
        'demo_reference_leaks':[r['id'] for r in data['rows'] if any(s.startswith('DEMO/') for seg in r['structured']['segments'] for s in seg['source_ids'])]}
    changes={}
    for variant in ['control','fewshot']:
        old={r['id']:r for r in prior['rows'] if r['variant']==variant and r['group']!='transfer8'}
        changes[variant]={'improved':[r['id'] for r in reviewed if r['id'] in old and not old[r['id']]['content_correct'] and r['content_correct']],
            'regressed':[r['id'] for r in reviewed if r['id'] in old and old[r['id']]['content_correct'] and not r['content_correct']]}
    result={'method':'Primary agent read all135 v2 answers and checked source support; no independent second reviewer or model grader.',
        'design':'Sequential development: new v2 calls compared with saved exp31 control/v1, same frozen questions and evidence. Not independent holdout.',
        'criteria':prior['criteria'],'limits':prior['limits'],'model':prior['model'],
        'answers_sha256':hashlib.sha256(raw).hexdigest(),'summary':summary,'changes':changes,'rows':reviewed}
    (OUT/'exp32_primary_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
