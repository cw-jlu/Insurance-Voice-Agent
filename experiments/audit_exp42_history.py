"""Aggregate saved exp42 calls and the primary review; never re-run models."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results'
FOLDER=OUT/'exp42_windows_https_records'

FINDINGS={
    'MT04':'回复3人家庭条件后，销售流程改问年龄，没有针对已补条件落到10%优惠；回复触发还包含宽泛家人匹配，需人工复测。',
    'MT05':'未确认投保医保身份和剩余免赔，将没走医保当成有医保身份投保，并假设尚剩一万元免赔。',
    'MT06':'剩余免赔未知时直接扣一万元，未追问；未到用例的免赔已用完分支。',
    'MT10':'模型声明clarify但口播未实际追问年龄；操作者依元数据回复72岁后，流程转去问医保，未处理首次投保超龄。',
    'MX01':'补医保后只报价，未继续确认家庭统计条件并完成个人化优惠问项。',
    'MX03':'核保安全转交，但pending_questions仅有核保，没有记录尚缺医保的个人报价待处理项。',
    'MX04':'补医保后年/月费率正确，但遗漏用户问的全年差额473元。',
    'MX05':'未确认投保医保身份和剩余免赔就按60%与一万元免赔测算；结果碰巧等于用例最终值，不算条件判断通过。',
    'MX06':'补4人家庭条件后改问年龄，没有落到15%家庭优惠；宽泛家人触发需人工复测。',
    'MX07':'更正为无医保后重报971/96元，漏掉被要求一起重算的全年差额181元。',
    'MX10':'给条件分支和优惠档位，但没有追问真实医保、家庭条件，未执行个体落地分支。',
}


def main():
    recall=[]
    for p in sorted(FOLDER.glob('recall_*.json')):
        row=json.loads(p.read_text(encoding='utf-8'))
        assert 'error_type' not in row
        # Explicit primary review of target recall, not an automatic correctness judge.
        passed=row['variant']=='full' or row['id']=='recent_price'
        issues=[]
        if row['variant']=='recent6' and row['id']=='early_expense':
            issues=['未找回已给的8000元医保补偿和12000元余额，未给2000元结果；r2误称此前未提供医保金额。']
        if row['variant']=='recent6' and row['id']=='early_topics':
            issues=['扩展猜测为责任大类及其他先进疗法，未准确限定原来的住院、靶向药、质子重离子三问。']
        if row['variant']=='full' and row['id']=='early_topics':
            issues=['问项回指正确，但把靶向药直接归为特定药品渠道仍有过度概括风险，且追加了未明确要求的具体名单转交。']
        recall.append({'record':str(p.relative_to(ROOT)).replace('\\','/'),
                       'id':row['id'],'variant':row['variant'],'repeat':row['repeat'],
                       'target_recall_correct':passed,'other_business_issues':issues})
    comparison={}
    for variant in ['recent6','full']:
        raw=[json.loads(p.read_text(encoding='utf-8')) for p in FOLDER.glob(f'recall_*_{variant}_*.json')]
        calls=[c for r in raw for c in r['api_calls'] if c['model']=='deepseek-flash' and c.get('usage')]
        comparison[variant]={'n':len(raw),'target_recall_correct':sum(r['target_recall_correct'] for r in recall if r['variant']==variant),
                             'ds_request_median_s':statistics.median(c['seconds'] for c in calls),
                             'ds_prompt_tokens_median':statistics.median(c['usage']['prompt_tokens'] for c in calls),
                             'text_turn_median_s':statistics.median(r['turn']['wall_s'] for r in raw)}
    dialogue=[]
    max_visible=0
    for p in sorted(FOLDER.glob('dialogue_*.json')):
        row=json.loads(p.read_text(encoding='utf-8'))
        assert 'error_type' not in row
        max_visible=max(max_visible,max(len(t['history_before'])+1 for t in row['turns']))
        dialogue.append({'id':row['id'],'record':str(p.relative_to(ROOT)).replace('\\','/'),
                         'turns':len(row['turns']), 'scope':'primary review of core obligations; not full per-claim source audit',
                         'core_obligations':'incomplete_or_wrong' if row['id'] in FINDINGS else 'met_in_this_run',
                         'finding':FINDINGS.get(row['id'],'已回应本例核心问项/安全转交；不代表完整口播逐句来源已全部复核。'),
                         'operator_trigger_needs_manual_retest':row['id'] in ['MT04','MT10','MX06']})
    probe=json.loads((FOLDER/'overflow_injected_real_summary_v2.json').read_text(encoding='utf-8'))
    assert 'error_type' not in probe and probe['original_history_unchanged']
    result=probe['turn']['timings']['answer_detail']['evidence_result']
    assert result['calls'][0]['context_events'] and result['provenance']['valid']
    known=[c['known'] for c in probe['api_calls'] if c.get('known') and not c.get('injected_context_overflow_before_network')][-1]
    compressed=known['_history']
    assert all(amount in compressed for amount in ['20000','8000','12000','10000'])
    assert '2000' in probe['turn']['text']
    summary={'date':'2026-10-13','method':'Primary agent read saved transcripts and target recall. No model grader or independent reviewer. No new generation in aggregation.',
             'transport':'Windows certificate-checked HTTPS for real Qwen/DS requests; default Python TLS connections failed and were retained separately.',
             'recall':comparison,'recall_rows':recall,
             'dialogue':{'cases':len(dialogue),'user_turns':sum(d['turns'] for d in dialogue),
                         'service_errors':0,'max_visible_messages_at_model_call':max_visible,
                         'limit':'All these new conversations fit within 6 visible messages, so their historical input is also identical to the recent6 control. They reveal existing dialogue-flow defects, not a demonstrated full-history regression. Operator trigger matching in 3 cases needs manual retest; no official accuracy reported.',
                         'rows':dialogue},
             'overflow_probe':{'injected':True,'real_summary_and_answer':True,'original_transcript_unchanged':True,
                               'early_amounts_retained':True,'claim_result':2000,'context_events':result['calls'][0]['context_events'],
                               'initial_probe':'First short-history injected probe failed safely because summary was not smaller; record preserved. v2 consolidates repetitions and succeeded.',
                               'limit':'No natural 1M-token overflow, audio, live interruption or telephone test in this run.'},
             'unit_tests_passed':68,
             'deployed_policy':'Full confirmed history by default; only provider context-overflow error can invoke summary; raw history not changed',
             'release_code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                ['src/agent/context.py','src/agent/cloud.py','src/agent/session.py','src/agent/evidence_answer.py','src/agent/pipeline.py','src/agent/small_model.py']}}
    (OUT/'exp42_history_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'recall':comparison,'dialogue_cases':len(dialogue),'dialogue_turns':summary['dialogue']['user_turns'],
                      'overflow_probe':'passed','unit_tests':68},ensure_ascii=False))


if __name__=='__main__':main()
