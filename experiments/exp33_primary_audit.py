"""Aggregate recorded primary-agent judgments; no text-based automatic grader."""
import hashlib
import json
import math
import statistics
from pathlib import Path

OUT=Path(__file__).resolve().parent/'results'
FOCUS=['C09','C11','C15','C19','C21','Y002','Y022','Y033','Y034','E08','E22','E28']

def main():
    raw=(OUT/'exp33_minimum_bound.json').read_bytes()
    data=json.loads(raw)
    decisions=json.loads((OUT/'exp33_manual_decisions.json').read_text(encoding='utf-8'))
    assert len(data['rows'])==len(decisions)==350
    def key(r):return f"{r['variant']}_{r['id']}_r{r['round']}"
    assert {key(r) for r in data['rows']}==set(decisions)
    reviewed=[]
    for r in data['rows']:
        c,t,reason=decisions[key(r)]
        assert not t or r['provenance']['valid']
        reviewed.append({**{k:r[k] for k in ['id','variant','round','group','trial_group','question']},
            'content_correct':c,'traceable':t,'joint_correct':c and t,'reason':reason,'answer':r['text'],
            'record_sha256':hashlib.sha256((OUT/'exp33_records'/(key(r)+'.json')).read_bytes()).hexdigest()})
    def metrics(rs):
        n=len(rs)
        return {'n':n,**{k:{'passed':sum(r[k] for r in rs),'percent':round(100*sum(r[k] for r in rs)/n,2)} for k in ['content_correct','traceable','joint_correct']},
            'content_failures':[f"{r['id']}/r{r['round']}" for r in rs if not r['content_correct']],
            'trace_failures':[f"{r['id']}/r{r['round']}" for r in rs if not r['traceable']]}
    summary={}
    for v in ['v2','v3_organized']:
        rs=[r for r in reviewed if r['variant']==v]
        original=[r for r in data['rows'] if r['variant']==v and r['trial_group']=='original127']
        ts=sorted(r['total_s'] for r in original)
        focus=[r for r in rs if r['id'] in FOCUS]
        new=[r for r in rs if r['trial_group']=='new12']
        summary[v]={'original127':metrics([r for r in rs if r['trial_group']=='original127']),
            'original_groups':{g:metrics([r for r in rs if r['trial_group']=='original127' and r['group']==g]) for g in ['development71','prior_boundary16','new40']},
            'focus12_three_rounds':metrics(focus),'new12_two_rounds':metrics(new),
            'focus_per_question':{i:{'content':sum(r['content_correct'] for r in focus if r['id']==i),'joint':sum(r['joint_correct'] for r in focus if r['id']==i),'rounds':3} for i in FOCUS},
            'new_per_question':{i:{'content':sum(r['content_correct'] for r in new if r['id']==i),'joint':sum(r['joint_correct'] for r in new if r['id']==i),'rounds':2} for i in sorted({r['id'] for r in new})},
            'latency_full_text_s':{'median':statistics.median(ts),'p95':ts[math.ceil(.95*len(ts))-1],'max':max(ts)},
            'median_chars':statistics.median(len(r['text']) for r in original),
            'mechanical_provenance_pass':sum(r['provenance']['valid'] for r in original),
            'call_errors':[key(r) for r in data['rows'] if r['variant']==v and r.get('error_type')],
            'demo_leaks':[key(r) for r in data['rows'] if r['variant']==v and any(sid.startswith('DEMO/') for seg in r['structured']['segments'] for sid in seg['source_ids'])]}
    control={r['id']:r for r in reviewed if r['variant']=='v2' and r['trial_group']=='original127'}
    candidate={r['id']:r for r in reviewed if r['variant']=='v3_organized' and r['trial_group']=='original127'}
    result={'method':'Primary agent read all350 answers and checked source support. No independent second reviewer or model grading.',
        'design':json.loads((OUT/'exp33_plan.json').read_text(encoding='utf-8'))['design'],
        'criteria':'Whole answer correctness: necessary branches preserved, extra false conditions fail; valid IDs do not imply semantic support. Scope claims require inventory or directly bounded local evidence.',
        'limits':['Fixed development questions and synthetic diagnostic new questions, not random independent customer holdout.',
            'Temperature0; repeated calls can be identical and are correlated, not independent sample counts.',
            'E09/E19 retain personal residual expense interpretation; question ambiguity remains.',
            'Prompt and source organization change together; contribution cannot be isolated.',
            'Text completion timing is not audio first response; STT/TTS not invoked.'],
        'answers_sha256':hashlib.sha256(raw).hexdigest(),'summary':summary,
        'improved_content':[i for i in control if not control[i]['content_correct'] and candidate[i]['content_correct']],
        'regressed_content':[i for i in control if control[i]['content_correct'] and not candidate[i]['content_correct']],
        'rows':reviewed}
    (OUT/'exp33_primary_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
