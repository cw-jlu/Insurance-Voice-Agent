"""Freeze v2 after reviewing v1; reuse explicitly named recent control snapshots."""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from agent.evidence_answer import Evidence, EvidenceAnswerer, fewshot_system
from agent import cloud
from exp30_fewshot import save

OUT=ROOT/'experiments/results'

def main():
    folder=OUT/'exp32_records'
    folder.mkdir(exist_ok=True)
    prior=(OUT/'exp31_plan.json').read_bytes()
    previous=json.loads(prior)
    sources=(OUT/'exp29_sources.json').read_bytes()
    plan={'cases':previous['cases'],'prompt':fewshot_system('v2'),'model':cloud.SLOW_MODEL,
        'source_sha256':hashlib.sha256(sources).hexdigest(),
        'comparison_plan_sha256':hashlib.sha256(prior).hexdigest(),
        'design':'135 new v2 single-pass calls; compare saved exp31 fresh control and v1. Sequential prompt development, not independent held-out or simultaneous three-arm trial.'}
    path=OUT/'exp32_plan.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==plan,'Frozen v2 plan changed'
    else:
        save(path,plan)
    evidence=[Evidence(**e) for e in json.loads(sources)['evidence']]
    answerer=EvidenceAnswerer()
    def one(case):
        row={**case,'variant':'fewshot_v2'}
        try:
            row.update(answerer.answer(case['question'],case['known'],evidence,fewshot=True,fewshot_version='v2'))
        except Exception as exc:
            row['error_type']=type(exc).__name__
        return row
    with ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(one,c) for c in plan['cases'] if not (folder/f"{c['id']}.json").exists()]
        for job in as_completed(jobs):
            row=job.result();save(folder/f"{row['id']}.json",row)
            print(json.dumps({k:row.get(k) for k in ['id','total_s','error_type']},ensure_ascii=False),flush=True)
    rows=[json.loads((folder/f"{c['id']}.json").read_text(encoding='utf-8')) for c in plan['cases']]
    save(OUT/'exp32_fewshot_v2.json',{'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':rows})

if __name__=='__main__':
    main()
