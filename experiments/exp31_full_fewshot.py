"""Frozen full paired comparison: 127 original cases plus 8 transfer cases."""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from agent.evidence_answer import Evidence, EvidenceAnswerer, BOUND_SYSTEM, fewshot_system
from agent import cloud
from exp30_fewshot import NEW, save

OUT=ROOT/'experiments/results'

def main():
    folder=OUT/'exp31_records'
    folder.mkdir(exist_ok=True)
    cases=json.loads((OUT/'exp29_questions.json').read_text(encoding='utf-8'))['cases']
    cases += [{'id':i,'question':q,'known':{},'group':'transfer8'} for i,q in NEW]
    sources=(OUT/'exp29_sources.json').read_bytes()
    plan={'cases':cases,'prompts':{'control':BOUND_SYSTEM,'fewshot':fewshot_system()},
          'model':cloud.SLOW_MODEL,'source_sha256':hashlib.sha256(sources).hexdigest(),
          'design':'135 questions x 2 fresh single-pass samples. Primary comparison on unchanged original127; transfer8 reported separately. No repair.'}
    path=OUT/'exp31_plan.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==plan,'Frozen plan changed'
    else:
        save(path,plan)
    answerer=EvidenceAnswerer()
    evidence=[Evidence(**e) for e in json.loads(sources)['evidence']]
    def one(variant,case):
        row={**case,'variant':variant}
        try:
            row.update(answerer.answer(case['question'],case['known'],evidence,
                bound=variant=='control',fewshot=variant=='fewshot'))
        except Exception as exc:
            row['error_type']=type(exc).__name__
        return row
    with ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(one,v,c) for c in cases for v in plan['prompts']
              if not (folder/f"{v}_{c['id']}.json").exists()]
        for job in as_completed(jobs):
            row=job.result()
            save(folder/f"{row['variant']}_{row['id']}.json",row)
            print(json.dumps({k:row.get(k) for k in ['variant','id','total_s','error_type']},ensure_ascii=False),flush=True)
    rows=[json.loads((folder/f"{v}_{c['id']}.json").read_text(encoding='utf-8')) for c in cases for v in plan['prompts']]
    save(OUT/'exp31_full_fewshot.json',{'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':rows})

if __name__=='__main__':
    main()
