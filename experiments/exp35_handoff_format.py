"""Fresh v2 decisions on exp34 frozen cases; reuse saved control explicitly."""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments')]
from agent.evidence_answer import Evidence, EvidenceAnswerer, handoff_system
from exp30_fewshot import save

def main():
    out = ROOT/'experiments/results'
    prior = json.loads((out/'exp34_plan.json').read_text(encoding='utf-8'))
    source_raw = (out/'exp33_organized_sources.json').read_bytes()
    assert hashlib.sha256(source_raw).hexdigest() == prior['source_sha256']
    evidence = [Evidence(**e) for e in json.loads(source_raw)['evidence']]
    plan = {**prior, 'prompts': {'post_handoff_v2': handoff_system('v2')},
        'design': '143 fresh v2 one-pass calls. Compare sequentially with saved exp34 control; no resampling of exp34 errors. Same cases, policies and evidence. Primary-agent audit only.'}
    path = out/'exp35_plan.json'
    folder = out/'exp35_records'
    folder.mkdir(exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == plan
    else:
        save(path, plan)
    engine = EvidenceAnswerer()
    def one(c):
        row = {**c, 'variant': 'post_handoff_v2'}
        try:
            row.update(engine.answer(c['question'], c['known'], evidence,
                fewshot=True, fewshot_version='v3', allow_handoff=True, handoff_version='v2'))
        except Exception as exc:
            row['error_type'] = type(exc).__name__
        return row
    with ThreadPoolExecutor(4) as pool:
        jobs = [pool.submit(one,c) for c in plan['cases'] if not (folder/f"{c['id']}.json").exists()]
        for job in as_completed(jobs):
            row = job.result()
            save(folder/f"{row['id']}.json", row)
            print(json.dumps({k:row.get(k) for k in ['id','total_s','error_type','outcome_validation','outcome']},ensure_ascii=False),flush=True)
    rows = [json.loads((folder/f"{c['id']}.json").read_text(encoding='utf-8')) for c in plan['cases']]
    save(out/'exp35_handoff_format.json', {'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':rows})

if __name__ == '__main__':
    main()
