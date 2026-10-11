"""Record the primary agent's manual review; this is not an automatic judge.

Decisions were made after reading the frozen answers and their evidence. Running
this file only records those decisions and computes counts, without model calls.
"""
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from exp28_primary_audit import NOTES as PRIOR_NOTES

OUT = Path(__file__).resolve().parent / 'results'
NOTES = dict(PRIOR_NOTES)
NOTES.update({
 'C07':'本轮转诊及护工都明确合同100种重疾，转院证明、指定服务及结算正确。',
 'C15':'本轮保留牙科/屈光意外例外，一般医疗及身份60%限定正确。',
 'Y012':'直接核查原始xlsx D118：ICU本身也有同病因间隔30日合并住院规定，回答正确；不按初步误读扣分。',
 'Y015':'本轮同时说明特需与先进疗法两项床位1500及房型限制，限定合同疾病/医院，未套用基础。',
 'Y036':'本轮保留牙科意外例外。',
 'Y038':'已知慢性胃炎费用免责与核保区分，不承诺承保。',
 'N01':'318年缴、31月缴，372-318=54；两个费率行齐全。',
 'N02':'31岁423/42，504-423=81；年龄边界及标准费率条件正确。',
 'N03':'无社保身份不满足有医保身份投保未结算的60%合取条件。',
 'N04':'15000扣未使用1万，再乘60%=3000，假设完整。',
 'N05':'骨折意外与妊娠原因费用区分，医保身份条件和医院范围正确。',
 'N06':'家庭共享只一般医疗；普通门急诊不计家庭共享，与本人两责任共享区分。',
 'N07':'见错误判定：已知既往结节仍先承诺20%，且把60%当覆盖比例。',
 'N08':'见错误判定：满足年龄不等于符合整体投保条件。',
 'N09':'见错误判定：明确指定期限重新投保仍套首次55岁限制。',
 'N10':'明确主体1万与服务不限额冲突，区分服务、医院及不合格机构路径，未承诺一概不限额。',
 'N11':'一般700+ICU1500=2200，另购已知、医院和疾病等待/意外条件正确。',
 'N12':'意外不受疾病30天等待；一般3天与ICU0免赔区分正确。',
 'N13':'已知满足合同重疾，不套一般免赔；重疾0及身份60%正确。',
 'N14':'自己找护工不符合指定服务、机构直结条件，不承诺发票报销。',
 'N15':'不承诺105岁保证续保，881费率正确；未知结论没有来源范围引用。',
 'N16':'正好6岁不能开处方；指定平台50%、月100/年1000和额度假设正确。',
 'E01':'4岁有社保806，0—4闭区间正确。',
 'E02':'5岁有社保279，5—10闭区间正确。',
 'E03':'55岁无社保月399及家庭价条件正确。',
 'E04':'56岁有社保年1645及标准费率条件正确。',
 'E05':'56元女性费率正确；额外断言等待30天才给付，遗漏D99意外路径。',
 'E06':'31岁男性51元，未混入基础保费。',
 'E07':'8000低于未使用1万年免赔，赔0，不给负数。',
 'E08':'医保已报10000仍漏扣，详见错误记录。',
 'E09':'年免赔已用6000仍重扣完整10000，详见错误记录。',
 'E10':'无社保身份15000减10000=5000，不强套60%。',
 'E11':'两天低于次免赔3天给0，未负数赔付。',
 'E12':'一般0加ICU1000，未把一般免赔套ICU；“2减3等于0天”按可给付天数零下限理解，措辞不严谨但结果正确。',
 'E13':'40减3=37给付天，再限30，3000，不先截住院天数再减免赔。',
 'E14':'同病因间隔20日合并，只扣一次3天，1700。',
 'E15':'200合规费50%=100，单次及年度限额正确。',
 'E16':'有医保身份未结算30%，200*30%=60，条件正确。',
 'E17':'月剩10，50%计算50受剩余额度约束，赔10。',
 'E18':'400*50%=200，月/年剩余同时100，赔100。',
 'E19':'10000-7000错算1000，详见错误记录。',
 'E20':'指定民营普通部在基础范围，不一概私立不赔，身份条件正确。',
 'E21':'明确名单外不符合医院范围，随后可能无法/审核用语偏弱但没有承诺符合。',
 'E22':'无关车祸不按既往体质全面排除，费用原因范围与核保区分。',
 'E23':'同一投前疾病不能靠等待期解除既往症免责。',
 'E24':'非既往结节且医保已结，住院补偿20%=2000；额外无结节假设冗余但分支事实正确。',
 'E25':'完整R12只列20/60未给组合算法，明确不选择12/20/60；局部资料不足可由完整责任包核查，不代表所有正式条款都没规则。',
 'E26':'补偿包脊椎排除不自动跨到基础责任；已知非既往仍反问既往不必要，但未否认已知事实或承诺错误赔付。',
 'E27':'16康复与100重疾定义分开，不从前者推后者免赔。',
 'E28':'普通病不满足重大转诊疾病前提。',
 'E29':'仅一名亲属，年1万/标准房500与舱位上限正确。',
 'E30':'商务舱不能按其价格全额报，经济含超级经济舱；未编造全部不赔。',
 'E31':'未见基础床位1500不能证明基础无床位上限，详见错误记录。',
 'E32':'未编造药名，清单名称及更新规则正确；全资料没有完整清单的否定没有范围证据。',
 'E33':'未知100种完整定义，不凭原位癌名称判断；全资料缺定义的否定引用不具集合范围证明。',
 'E34':'未编完整民营名单、范围正确；资料不完整及保险公司最新名单说法缺直接集合/更新证据。',
 'E35':'外购药械需释义27特定条件，不把处方当充分条件；医院及海南扩展来自原文。',
 'E36':'105费率不是保证续保，引用一年期与指定期限重新投保规则支持推论，不作全资料缺规则断言。',
 'E37':'3吨非营运不满足4吨及以上营运条目；未承诺一定保。',
 'E38':'危货运输在排除职业表，不用年龄代替职业资格；因此导致费用免责。',
 'E39':'71首投超70，71费率是指定期限重新投保，不保证首投。',
 'E40':'不同投保人/订单不能跨单共享，且家庭共享不含一般门急诊。',
})

# Manual exceptions: facts, coverage of required answer, conditions, tracing,
# observed failure category, and source-grounded reason. All other rows passed.
ISSUES = {
 'C05':(True,False,True,True,'必要分支遗漏','只说一般住院扣1万，未说明符合合同重疾时无免赔，不能只提可能属于重大疾病却不回答其给付区别。'),
 'C09':(True,False,False,True,'未应用用户已知事实','用户已有高血压，仍转到如果不是既往症，没有直接落实当前病史的费用边界；引用条款自身正确。'),
 'C12':(False,True,False,False,'比例条件组合未定义','结节20%后声称医保未结两责任都会调60%；R12:D107/D108未明确同时满足的优先级，不能承诺60%覆盖20%。'),
 'C19':(True,False,False,False,'未呈现原资料范围冲突','只讲服务100不限额/医院60一万，遗漏主体保险D49一万与服务D52不限额差异，不能由服务备注直接概括保险给付。'),
 'C20':(False,False,False,False,'未应用用户已知事实','投前已有结节，却再说另购补偿包结节20%，没有把这笔已有结节相关手术明确排除，易造成加包可覆盖印象。'),
 'C22':(True,True,True,False,'排他结论缺依据','各路径金额和限制正确；仅列R04不能充分证明所有康复相关救护都只有此责任覆盖，正式费用定义未齐。此项只扣严格溯源，不认定已证实赔付错误。'),
 'Y026':(True,True,True,False,'资料缺失断言无范围证据','首次70及未知最高续保年龄结论合理，局部投保年龄/费率引用不能证明全资料没有最高年龄规定。'),
 'Y034':(True,True,True,False,'资料缺失断言无范围证据','69岁3316正确；无69岁续保价的集合否定缺范围证明，额外71岁假设也不必要。'),
 'Y040':(True,True,True,False,'资料缺失断言无范围证据','不保证续保和一年期正确；资料没直接续保规则的全局否定未引用资料目录/缺失记录。'),
 'N07':(False,False,False,False,'未应用用户已知事实','先说结节补偿确实20%，后才说既往相关不赔且再留认定悬念；用户已明确投前查出同一结节，另有20/60组合覆盖问题。'),
 'N08':(False,False,False,False,'整体资格前后矛盾','先因42岁说可以投保，后说4吨营运货车在排除表可能不能；年龄是必要而非充分条件，职业事实已明确。'),
 'N09':(False,False,False,False,'错误套用首次年龄限制','用户已明确指定期限重新投保；首句仍因56超过首次55拒绝，第二句却正确引56—60费率1019，内在矛盾。'),
 'N15':(True,True,True,False,'资料缺失断言无范围证据','不保证105岁与881价格正确；第一段资料没有支持续保条款未给任何source_ids。'),
 'E05':(True,True,False,False,'额外说明缩窄条件','价格56正确；额外说需等待30天初诊才给付，原始D99还允许意外导致罹患，不能用才排除该分支。'),
 'E08':(False,False,False,False,'计算漏扣医保补偿','回答10000，正确为max(20000-10000医保已报-10000未用年免赔,0)=0；免责#2已在所引证据中，模型公式却没应用。'),
 'E09':(False,False,False,False,'未应用年度累计状态','回答0，正确剩余免赔10000-6000=4000，按本题个人合规费用口径9000-4000=5000；若9000含未量化医保报销须另扣，不能径直重新扣全年度1万。'),
 'E19':(False,False,False,False,'算术及年度累计状态','说已抵7000共享正确，却把10000-7000算成1000，给4000；本题个人合规费用口径应剩3000、赔2000。若5000含医保已报须另扣，该口径歧义不影响现有减法错误。'),
 'E31':(False,True,False,False,'未记载被推成无约束','基础摘要未列1500，不能断言基础没有任何1500限制；完整合同住院费用定义未提供。应说现有基础材料未见，不能把另两责任上限迁移。'),
 'E32':(True,True,True,False,'资料缺失断言无范围证据','不猜清单正确；清单名称及更新条款不能证明当前整个资料包不存在完整明细。'),
 'E33':(True,True,True,False,'资料缺失断言无范围证据','保留未知正确；责任文字引用不证明所有附件均无定义/病种表。'),
 'E34':(True,True,True,False,'资料缺失断言无范围证据','不编医院名单正确；医院范围引用不证明全包缺名单，也不证明民营名单以最新公布信息为准。'),
}

def main():
    path=OUT/'exp29_expanded_locations.json'
    raw=path.read_bytes()
    data=json.loads(raw)
    reviewed=[]
    for row in data['rows']:
        qid=row['id']
        assert qid in NOTES, qid
        f,c,k,t,category,reason=ISSUES.get(qid,(True,True,True,True,'',NOTES[qid]))
        reviewed.append({
            'id':qid,'group':row['group'],'question':row['question'],
            'facts_correct':f,'complete':c,'conditions_ok':k,
            'content_correct':f and c and k,'traceable':t,
            'joint_correct':f and c and k and t,
            'error_stage':'answer_generation_or_claim_support' if category else None,
            'error_category':category or None,'reason':reason,'review_note':NOTES[qid],
            'answer':row['text'],'source_ids':list(dict.fromkeys(
                sid for s in row.get('structured',{}).get('segments',[]) for sid in s['source_ids'])),
            'mechanical_provenance':row.get('provenance'),
            'record_sha256':hashlib.sha256((OUT/'exp29_records'/f'{qid}.json').read_bytes()).hexdigest(),
        })
    assert len(reviewed)==127 and len({r['id'] for r in reviewed})==127
    def metrics(rs):
        n=len(rs)
        return {'n':n,**{key:{'passed':sum(r[key] for r in rs),
            'percent':round(100*sum(r[key] for r in rs)/n,2)}
            for key in ['content_correct','traceable','joint_correct']},
            'content_failures':[r['id'] for r in rs if not r['content_correct']],
            'trace_failures':[r['id'] for r in rs if not r['traceable']]}
    times=sorted(r['total_s'] for r in data['rows'])
    result={
        'method':'Primary agent manually read every frozen answer and reviewed against source; no API judge, no independent second reviewer.',
        'criteria':'Content requires factual correctness, required branches and conditions. Strict tracing additionally requires citations to support each assertion, including collection-wide absence/exclusivity. Mechanical valid IDs/quotes are not semantic tracing.',
        'limits':['Fixed development plus deliberately selected diagnostic cases, not independent random customer holdout.',
                  'Single frozen sample per question; no answer repair or prompt changes.',
                  'E09/E19 reimbursed expense not numerically specified: reference amount assumes stated compliant expense is personal residual expense; wrong remaining annual deductible is unambiguous.',
                  'Latency is full text completion, not first audio; STT/TTS not called.',
                  'Mechanisms are inferred from observable answers, not model internal reasoning.'],
        'model':data['model'],'answer_snapshot_sha256':hashlib.sha256(raw).hexdigest(),
        'source_snapshot_sha256':hashlib.sha256((OUT/'exp29_sources.json').read_bytes()).hexdigest(),
        'groups':{g:metrics([r for r in reviewed if r['group']==g]) for g in ['development71','prior_boundary16','new40']},
        'overall':metrics(reviewed),
        'latency_s':{'median':statistics.median(times),'p95_nearest_rank':times[120],'max':max(times)},
        'median_answer_chars':statistics.median(len(r['text']) for r in data['rows']),
        'mechanical_provenance_pass':sum(r.get('provenance',{}).get('valid',False) for r in data['rows']),
        'model_calls':sum(len(r['calls']) for r in data['rows']),
        'retained_empty_answers':sum(not r['text'] for r in data['rows']),
        'content_error_categories':dict(Counter(r['error_category'] for r in reviewed if not r['content_correct'])),
        'rows':reviewed,
    }
    (OUT/'exp29_primary_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
