"""Expand declared cell IDs with parent context; preserve frozen model outputs."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agent.evidence_answer import Evidence, source_registry, citation_check
OUT=ROOT/'experiments/results'

def main():
    path=OUT/'exp28_grouped.json'
    result=json.loads(path.read_text(encoding='utf-8'))
    snapshot=json.loads((OUT/result['snapshot']).read_text(encoding='utf-8'))
    evidence=[Evidence(**e) for e in snapshot['evidence']]
    registry=source_registry(evidence)
    result['rendering']={
        'frozen_output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'note':'Only canonical citation rendering corrected for declared nested cell IDs; model calls, text, source_ids, timing unchanged. No generation rerun.',
        'aliases':{sid:next(e.id for e in evidence if e.text==ref.text) for sid,ref in registry.items() if sid not in {e.id for e in evidence}}}
    for row in result['rows']:
        if row.get('error_type'):
            continue
        structured=row['structured']
        structured['claims']=[{**s,'citations':[{'id':sid,'quote':registry[sid].text if sid in registry else ''}
                                               for sid in s.get('source_ids',[])]}
                              for s in structured['segments']]
        assert row['text']==''.join(s['text'] for s in structured['segments'])
        row['provenance']=citation_check(structured,evidence)
    (OUT/'exp28_grouped_rendered.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('rows',len(result['rows']),'valid provenance',sum(r.get('provenance',{}).get('valid',False)for r in result['rows']))

if __name__=='__main__':main()
