"""Aggregate frozen outputs and recorded source reviews; no model scoring."""
import hashlib
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'results'

def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))

def main():
    # Preserve independent reviewers' original verdicts. Apply the same concise-answer
    # rubric as the primary review: inherited conditions need not be repeated;
    # bounded conditional answers need not invent an unknown user profile.
    adjudications = [
        {'variant': 'evidence_full', 'id': 'C02', 'overall': True,
         'reason': '逐项回答住院、特药和质子重离子及免赔；明确合同定义、指定路径、审核及大陆限制。本题未问全部等待/社保分支，未复述未问全部条件不单独判失败。逐句引用仍不足，traceable保留false。'},
        {'variant': 'evidence_full', 'id': 'C17', 'overall': True,
         'reason': '有社保身份投保但未结算条件已在同一回答首段明确，后文可承接，不要求每段重复；引用不足的判定保留。'},
        {'variant': 'evidence_full', 'id': 'Y023', 'overall': True,
         'reason': '用户未明确投保社保身份，回答以有社保身份投保明确条件化60%，未把无社保身份强套60%；未直接给另一分支可改进，但有界回答不算错误承诺。引用不足保留。'},
    ]
    (OUT / 'exp27_adjudication.json').write_text(json.dumps(adjudications, ensure_ascii=False, indent=2), encoding='utf-8')
    specs = [
        ('legacy','exp27_legacy.json','exp27_control_audit.json'),
        ('prompt_sparse','exp27_evidence_quality_v1.json','exp27_sparse_audit.json'),
        ('evidence_full','exp27_evidence_quality_v1.json','exp27_full_audit.json'),
        ('debate_full','exp27_evidence_quality_v1.json','exp27_debate_audit.json'),
        ('minimal_full','exp27_minimal_full.json','exp27_minimal_audit.json'),
        ('bound_full','exp27_bound_full.json','exp27_bound_primary_audit.json'),
    ]
    summary = []
    for variant, output, audit in specs:
        rows = [r for r in read(output)['rows'] if r['variant'] == variant]
        reviews = read(audit)['rows']
        assert len(rows) == len(reviews) == 71
        original = sum(r['overall'] for r in reviews)
        reviews = {r['id']: dict(r) for r in reviews}
        for adj in adjudications:
            if adj['variant'] == variant:
                reviews[adj['id']]['overall'] = adj['overall']
        times = sorted(r['total_s'] for r in rows if not r.get('error_type'))
        summary.append(dict(variant=variant, n=71,
            independent_original_pass=original,
            content_pass=sum(r['overall'] for r in reviews.values()),
            trace_pass=None if variant=='legacy' else sum(r.get('traceable',False) for r in reviews.values()),
            joint_pass=None if variant=='legacy' else sum(r['overall'] and r.get('traceable',False) for r in reviews.values()),
            errors=sum(bool(r.get('error_type')) for r in rows),
            full_text_median_s=statistics.median(times),
            full_text_p90_s=times[int((len(times)-1)*.9)],
            median_chars=statistics.median(len(r.get('text','')) for r in rows),
            output=output, audit=audit,
            output_sha256=hashlib.sha256((OUT/output).read_bytes()).hexdigest()))
    (OUT/'exp27_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
