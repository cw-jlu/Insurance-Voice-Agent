"""Aggregate immutable v4 records; amount presence is not semantic correctness."""
import json
import re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=json.loads((ROOT/'experiments/data/complex_dialogue_v1.json').read_text(encoding='utf-8'))
    rows=[]
    counts=Counter()
    for case in data['cases']:
        folder='exp43_subject_fix_records' if case['id']=='MX09' else 'exp43_verified_records'
        path=ROOT/'experiments/results'/folder/(case['id']+'.json')
        record=json.loads(path.read_text(encoding='utf-8'))
        turns=record['turns']
        amount_checks=[]
        for oracle in case['numeric_oracles']:
            candidates=turns[:1] if oracle['after']=='initial_answer' else turns[-1:]
            amounts={float(x) for t in candidates for x in re.findall(r'\d+(?:\.\d+)?',t['text'].replace(',',''))}
            amount_checks.append({'expected':oracle['expected'],'checkpoint':oracle['after'],
                                  'present_in_speech':float(oracle['expected']) in amounts})
        ledger=turns[-1]['consultation']
        faults=[t['timings'].get('answer_detail',{}).get('consultation_fault') for t in turns]
        calls=[c for c in record['api_calls'] if c.get('usage')]
        counts.update({'cases':1,'turns':len(turns),ledger.get('status','no_consultation'):1,
                       'numeric_checks':len(amount_checks),
                       'numeric_present':sum(c['present_in_speech'] for c in amount_checks),
                       'schema_faults':sum(bool(f) for f in faults),
                       'network_errors':bool(record.get('error_type')),
                       'recorded_api_calls':len(calls)})
        rows.append({'id':case['id'],'record':path.relative_to(ROOT).as_posix(),
                     'final_state':ledger.get('status'),'task_states':[(t['id'],t['status']) for t in ledger.get('tasks',[])],
                     'numeric_presence_checks':amount_checks,'handoff_saved':bool(record.get('turns',[])[-1].get('handoff_request')),
                     'manual_scope_note':{
                        'MT05':'6600 and missing-condition collection correct; speech also says deduct annual 10000 before explaining remaining 4000; phrasing ambiguous.',
                        'MT06':'12000 and missing-condition collection correct; speech says remaining 12000 after deducting 8000, which ambiguously repeats the already-deducted settlement.',
                        'MX09':'Latest explicit replacement fix separately rerun; prior-person history stays, personal facts reset; current-person standard premium correct. Mother eligibility not explicitly restated.',
                     }.get(case['id'],'Core requested topics/conditions reviewed; not a full per-sentence source-entailment audit.')})
    summary={'scope':'Iterative development validation: 29 records from verified cohort plus the separately rerun insured-person replacement fix. No best-answer selection; all earlier cohorts preserved.',
             'counts':dict(counts),'unit_tests':90,'official_143_question_metrics_unchanged':True,
             'not_measured':['real voice first response/completion','live interruption','natural provider context overflow','new blind business accuracy'],
             'checks_limitation':'Final TODO state is a model/controller state. Numeric presence is only a diagnostic; neither proves semantic correctness or complete spoken coverage.',
             'cases':rows}
    (ROOT/'experiments/results/exp43_consultation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary['counts'],ensure_ascii=False))


if __name__=='__main__':main()
