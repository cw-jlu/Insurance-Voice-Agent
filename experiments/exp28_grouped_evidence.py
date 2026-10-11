"""Evidence-only vs evidence+scope prompt; freeze inputs, one call, no repairs."""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments')]
from agent import cloud
from agent.evidence_answer import EvidenceAnswerer, grouped_evidence, BOUND_SYSTEM, SCOPED_SYSTEM
from exp18_voice_prompt_ablation import rows_for

OUT = ROOT/'experiments/results'

def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1)

def main():
    path = OUT/'exp28_grouped.json'
    evidence = grouped_evidence()
    sources = {'evidence':[asdict(e) for e in evidence], 'files':[]}
    # Include immutable byte hashes of raw AND clean inputs, not only snippet hashes.
    for p in sorted((ROOT/'data/raw/尊享百万').glob('*')) + sorted((ROOT/'data/clean').rglob('*')):
        if p.is_file():
            sources['files'].append({'path':p.relative_to(ROOT).as_posix(),
                                     'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    if not (OUT/'exp28_grouped_sources.json').exists():
        save(OUT/'exp28_grouped_sources.json', sources)
    cases = rows_for('all')
    result = {'model':cloud.SLOW_MODEL, 'snapshot':'exp28_grouped_sources.json',
              'prompts':{'grouped_only':BOUND_SYSTEM,'grouped_scope':SCOPED_SYSTEM},
              'design':'71 frozen questions; original responsibility bundles; no reference answers, repair or retries',
              'rows':[]}
    # Resume an interrupted file write, never regenerate recorded answers or errors.
    for candidate in [path, path.with_suffix('.tmp')]:
        if candidate.exists():
            previous = json.loads(candidate.read_text(encoding='utf-8'))
            if len(previous['rows']) > len(result['rows']):
                result = previous
    done = {(r['id'],r['variant']) for r in result['rows']}
    answerer = EvidenceAnswerer()
    def one(case, variant):
        row = {'id':case['id'], 'variant':variant, 'question':case['question'],
               'known':dict(case.get('context') or {})}
        try:
            row.update(answerer.answer(row['question'], row['known'], evidence,
                                       bound=True, scoped=variant=='grouped_scope'))
        except Exception as exc:
            row['error_type'] = type(exc).__name__
        return row
    with ThreadPoolExecutor(4) as pool:
        jobs = [pool.submit(one, c, v) for c in cases for v in ['grouped_only','grouped_scope']
                if (c['id'],v) not in done]
        for f in as_completed(jobs):
            row = f.result()
            result['rows'].append(row)
            save(path, result)
            print(json.dumps({k:row.get(k) for k in ['id','variant','total_s','error_type']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
