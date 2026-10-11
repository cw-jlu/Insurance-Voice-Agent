"""Summarize recorded manual decisions; never infer correctness from answer text."""
import hashlib
import json
import math
import statistics
from pathlib import Path

OUT=Path(__file__).resolve().parent/'results'

def main():
    raw=(OUT/'exp31_full_fewshot.json').read_bytes()
    data=json.loads(raw)
    decisions=json.loads((OUT/'exp31_manual_decisions.json').read_text(encoding='utf-8'))
    assert len(data['rows'])==270 and len(decisions)==135
    reviewed=[]
    for row in data['rows']:
        ci,ti=(0,1) if row['variant']=='control' else (2,3)
        judgment=decisions[row['id']]
        reviewed.append({'id':row['id'],'variant':row['variant'],'group':row['group'],
            'content_correct':judgment[ci],'traceable':judgment[ti],
            'joint_correct':judgment[ci] and judgment[ti],
            'reason':judgment[4],'answer':row['text'],
            'mechanical_provenance':row['provenance'],
            'record_sha256':hashlib.sha256((OUT/'exp31_records'/f"{row['variant']}_{row['id']}.json").read_bytes()).hexdigest()})
    def metrics(rs):
        n=len(rs)
        return {'n':n,**{k:{'passed':sum(r[k] for r in rs),'percent':round(100*sum(r[k] for r in rs)/n,2)}
            for k in ['content_correct','traceable','joint_correct']},
            'content_failures':[r['id'] for r in rs if not r['content_correct']],
            'trace_failures':[r['id'] for r in rs if not r['traceable']]}
    summary={}
    for variant in ['control','fewshot']:
        rs=[r for r in reviewed if r['variant']==variant]
        original=[r for r in data['rows'] if r['variant']==variant and r['group']!='transfer8']
        times=sorted(r['total_s'] for r in original)
        summary[variant]={'original127':metrics([r for r in rs if r['group']!='transfer8']),
            'groups':{g:metrics([r for r in rs if r['group']==g]) for g in ['development71','prior_boundary16','new40','transfer8']},
            'latency_original127_s':{'median':statistics.median(times),'p95_nearest_rank':times[math.ceil(.95*len(times))-1],'max':max(times)},
            'median_chars_original127':statistics.median(len(r['text']) for r in original),
            'mechanical_provenance_pass_original127':sum(r['provenance']['valid'] for r in original),
            'demo_reference_leaks':[r['id'] for r in data['rows'] if r['variant']==variant and any(sid.startswith('DEMO/') for s in r['structured']['segments'] for sid in s['source_ids'])]}
    result={'method':'Primary agent read all270 answers and reviewed original sources and citations. Recorded manual content and semantic trace decisions; no API judge or independent second reviewer.',
        'criteria':'Entire answer must preserve source facts, necessary branches and conditions. Extra false restrictions fail even if headline amount/yes-no is correct. Source absence claims require scope evidence; IDs/quotes alone do not imply semantic support.',
        'limits':['Fixed development and diagnostic cases, not random independent customer holdout.',
                  'One retained answer per case/variant, no answer repair or score significance claim.',
                  'E09/E19 use personal residual expense interpretation; reimbursement amount ambiguity preserved.',
                  'Full text completion timing is not audio first response; no STT/TTS.',
                  'transfer8 reported separately, not added to main127 denominator.'],
        'model':json.loads((OUT/'exp31_plan.json').read_text(encoding='utf-8'))['model'],
        'answers_sha256':hashlib.sha256(raw).hexdigest(),'summary':summary,
        'improved_content':[i for i,d in decisions.items() if not i.startswith('F') and not d[0] and d[2]],
        'regressed_content':[i for i,d in decisions.items() if not i.startswith('F') and d[0] and not d[2]],
        'rows':reviewed}
    (OUT/'exp31_primary_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
