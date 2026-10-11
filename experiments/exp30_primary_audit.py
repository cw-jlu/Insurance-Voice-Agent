"""Primary agent's recorded manual content audit of 58 frozen pilot answers.

No model judge or automatic text correctness scoring. Citation format checks are
reported separately and must not be called semantic traceability.
"""
import hashlib
import json
import statistics
from pathlib import Path

OUT=Path(__file__).resolve().parent/'results'
# Explicit per-question judgments: baseline, fewshot, observed source-grounded note.
DECISIONS={
 'C05':(True,False,'对照同时给一般扣1万/合同重疾0免赔；fewshot只给一般医疗住院分支，必要分支遗漏，是本轮退步。'),
 'C09':(False,False,'问题说已有高血压、买了以后，属于当前病史在未来投保之前；两版仍绕回是否投前/新发，没有直接落实已知病史。'),
 'C12':(True,True,'两版都未再添加60%覆盖20%的组合规则；一般1万与补偿0免赔/结节20%分开，按非既往等费用合规条件语境理解。'),
 'C19':(False,False,'两版都讲主体检测1万/非指定60%，却漏服务D52不限额与主体D49的范围差异；fewshot有此行为示例仍未稳定迁移。'),
 'C20':(True,True,'两版都先说明既往相关手术不赔，并明确加包也受既往免责；未把20%说成解除免责。'),
 'N07':(False,True,'对照先说确实20%后再让已知投前病史待认定；fewshot直接按已知投前同病说明不能20%。'),
 'N08':(False,True,'对照先年龄可以、后职业可能不行；fewshot直接明确4吨营运在排除表，年龄不是充分条件。'),
 'N09':(False,True,'对照把问费率适用误读为费率是否不变，仍把已知指定期限当待确定条件；fewshot直接应用56+指定期限重投分支。'),
 'E05':(True,True,'女性31岁56元都正确。对照额外描述表中30天等待及合同疾病定义，但没有上一轮才给付的排他断言，本轮不按上一轮结果扣分；fewshot只报价格。'),
 'E08':(False,True,'对照10000，漏扣已报医保；fewshot20000-10000医保-10000免赔=0，引用R01/免责#2。'),
 'E09':(False,True,'对照先说0又说5000，前后矛盾；fewshot剩余免赔4000、赔5000。费用口径歧义沿用exp29说明。'),
 'E19':(True,True,'两版本轮都算剩余3000、赔2000；不能用exp29错算1000的样本作为本轮对照。'),
 'E31':(True,True,'两版本轮均把否定范围放在现有责任原文未列1500，未明确承诺没有任何床位限制；fewshot仍未清楚说明完整合同缺失，是表达/溯源待改善，不能把本轮局部未列直接当旧版无约束断言。'),
 'C01':(False,True,'对照分别说住院/门诊1万300万但未说共享，可能误导重复扣免赔；fewshot明确共享。'),
 'C07':(True,True,'两版合同100重疾、跨省转院证明、一亲属、1万标准房500及指定护工500/30/直结条件正确。fewshot重复最后总结更长。'),
 'Y012':(True,True,'两版ICU另购500/0/30/60及叠加正确；fewshot单次15000推导正确但非必答。'),
 'Y015':(False,True,'对照报特需1500及房型却漏必须合同重疾、需另购条件；fewshot补齐；没有用完整集合排他断言，只答特需病房路径。'),
 'Y036':(True,True,'两版牙科免责均保留意外例外。'),
 'N10':(True,True,'两版均呈现D49主体1万、D52服务不限额及D53医院60%一万/D54无资格路径。fewshot不同路径的解释仍应核对正式办理边界。'),
 'E13':(False,True,'对照先将住院40截为30再扣3，算2700；fewshot先扣3得到37再限给付30，3000。'),
 'E25':(True,True,'两版均承认当前完整责任包没有20/60组合规则，不编唯一比例。'),
 'F01':(False,True,'对照把已抵免赔6000当待扣6000，错给15000；fewshot剩余4000、32000-11000-4000=17000。'),
 'F02':(True,True,'两版剩余7500，本次个人6500不够免赔，赔0；对照负数为中间差额，最终没有给负数赔付。'),
 'F03':(True,True,'两版57指定期限重新投保费率档1019正确；fewshot另保留其他投保条件，不保证续保。'),
 'F04':(False,True,'对照先给囊肿20%再加60%覆盖，最后才既往不赔；fewshot直接说明同一投前囊肿不赔、加包不解除免责。'),
 'F05':(True,True,'两版将囊肿相关免责与无关车祸骨折区分，没有按疾病身份全面拒赔。'),
 'F06':(True,True,'两版34岁女性56元，均遵守明确只报价要求，无给付扩写。'),
 'F07':(True,True,'两版对具体床位上限保留未知；fewshot限定现有摘要未列，需完整合同定义，不承诺不限额。'),
 'F08':(False,False,'年龄不是充分条件说明正确，但用户已说明做危险货物运输作业，表中列明危险货物运输作业员；两版仍停在不能确认/需确认职业，而未明确这一已知匹配的资格不符。'),
}

def main():
    path=OUT/'exp30_fewshot.json'
    raw=path.read_bytes()
    data=json.loads(raw)
    reviewed=[]
    for row in data['rows']:
        baseline,fewshot,note=DECISIONS[row['id']]
        reviewed.append({'id':row['id'],'variant':row['variant'],'group':row['group'],
            'content_correct':baseline if row['variant']=='control' else fewshot,
            'reason':note,'answer':row['text'],
            'mechanical_provenance':row['provenance'],
            'record_sha256':hashlib.sha256((OUT/'exp30_records'/f"{row['variant']}_{row['id']}.json").read_bytes()).hexdigest()})
    assert len(reviewed)==58 and len(DECISIONS)==29
    summary={}
    for variant in ['control','fewshot']:
        rs=[r for r in reviewed if r['variant']==variant]
        def score(xs):
            passed=sum(r['content_correct'] for r in xs)
            return {'passed':passed,'n':len(xs),'percent':round(100*passed/len(xs),2),
                    'failures':[r['id'] for r in xs if not r['content_correct']]}
        original=[r for r in data['rows'] if r['variant']==variant]
        summary[variant]={'overall':score(rs),
            'groups':{g:score([r for r in rs if r['group']==g]) for g in ['prior_errors13','controls8','new8']},
            'median_total_s':statistics.median(r['total_s'] for r in original),
            'median_chars':statistics.median(len(r['text']) for r in original),
            'mechanical_provenance_pass':sum(r['provenance']['valid'] for r in original),
            'demo_reference_leaks':[r['id'] for r in original if any(sid.startswith('DEMO/') for s in r['structured']['segments'] for sid in s['source_ids'])]}
    result={'method':'Primary agent manual content review of 58 answers against source. No automatic/model judge, no independent second reviewer.',
        'limits':['Diagnostic sample intentionally enriched with prior failures; percentages are not general accuracy.',
                  'New8 were absent from examples but designed around the same patterns, not independent random held-out data.',
                  'Each variant one sample per question, no significance claim; no prompt revision after results.',
                  'Full text latency is not speech first response; cache-hit information was not measured.',
                  'No aggregate semantic traceability score for this pilot: mechanical provenance is not semantic support.',
                  'E09/E19 retain exp29 personal compliant expense assumption and reimbursement amount ambiguity.'],
        'model':json.loads((OUT/'exp30_plan.json').read_text(encoding='utf-8'))['model'],
        'answers_sha256':hashlib.sha256(raw).hexdigest(),'summary':summary,
        'improved':[i for i,(a,b,_) in DECISIONS.items() if not a and b],
        'regressed':[i for i,(a,b,_) in DECISIONS.items() if a and not b],
        'rows':reviewed}
    (OUT/'exp30_primary_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
