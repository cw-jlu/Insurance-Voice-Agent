"""Frozen fresh paired trial of v2 vs v3 + explicit source organization.

Original127 once, previous12 regressions three times, new12 twice.
Only descriptive results; new cases are synthetic, not independent customer data.
"""
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
FOCUS=['C09','C11','C15','C19','C21','Y002','Y022','Y033','Y034','E08','E22','E28']
NEW=[
 ('G01','我现在有胆囊息肉，下周买保险，半年后为这个息肉住院手术，住院补偿包能赔20%吗？'),
 ('G02','投保前有高血压，投保后在指定民营医院普通部因车祸骨折住院，骨折与高血压无关。会不会只因高血压就拒赔骨折？只解释这点。'),
 ('G03','我有重疾特需加油包，已满足合同重大疾病定义。保单指定民营医院VIP部和二级公立医院特需部，都在医院范围里吗？只说医院范围。'),
 ('G04','今天买了住院津贴包，明天车祸进ICU一天，所有其他条件满足。是不是必须等30天？这一天ICU津贴多少？'),
 ('G05','普通住院合规总费用28000元，医保已报9000元，年度免赔已抵扣3000元，没有其他补偿，以医保身份投保且已医保结算，一般医疗试算赔多少？'),
 ('G06','这次普通住院医保后个人合规费用16000元，年度免赔已抵扣4500元，没有其他补偿，以医保身份投保且已医保结算，一般医疗试算赔多少？'),
 ('G07','以无医保身份投保，恶性肿瘤重度院外特药已获审核、指定药店购买且在约定清单内，药在社保目录内，没有其他补偿。是不是也只能赔60%？'),
 ('G08','我67岁有医保，首次买必选计划，年缴标准保费是多少？只报价。'),
 ('G09','我67岁有医保，上一张保单期满后在指定期限内重新投保，资料能不能确定仍然年缴3316元？'),
 ('G10','给我完整的保单指定民营医院名单和合同100种重疾定义；如果本轮没有附件就说明缺什么，不猜。'),
 ('G11','恶性肿瘤重度基因检测，通过增值服务申请还是自己去合格机构检测，限额和结算方式一样吗？主体与服务备注不一致请说清。'),
 ('G12','满足合同重疾定义的住院化疗，与获批指定药店买的院外特药，都会统一扣1万元免赔吗？只比较免赔。'),
]

def organized_sources(base):
    extras=[]
    paths=sorted({e.path for e in base})
    manifest={'scope':'仅说明本轮注入资料范围，不是保险合同或承保承诺',
        'source_files':paths,'base_evidence_count':len(base),
        'included':['16项保障责任原文单元格及归属','投保须知逐段原文，含特殊职业表','责任免除逐段原文','年/月缴必选与加油包费率行、费率说明','系统身份配置'],
        'not_injected_as_documents':['完整保险合同及100种重大疾病逐项定义附件','完整约定药品清单附件','完整保单指定民营医院名单附件'],
        'boundary':'未注入附件不能证明真实合同不存在这些定义或限制；责任摘要没有列某限制不能推出合同不限额。'}
    text=json.dumps(manifest,ensure_ascii=False,indent=2)
    extras.append(Evidence('materials:manifest',text,'本轮资料目录（系统编制）','experiments/results/exp33_organized_sources.json',hashlib.sha256(text.encode()).hexdigest(),{'kind':'injected_materials_inventory','source_paths':paths}))
    for e in base:
        if e.id.startswith('rate-notice:') and e.id.endswith(':5'):
            text='【费率适用原文，分条展示，未改变文字】\n'+e.text.replace(';*',';\n*').replace('；*','；\n*')
            text+='\n【索引说明】年龄行不是独立的续投承诺；具体报价同时读取上述原文适用情形。高年龄的重投费率说明不代表只有高年龄才允许重新投保。未列明某年龄及时重投保费时，不能把首次或未及时重投行当成及时重投确定保费。'
            extras.append(Evidence('rate-scope:'+e.id.split(':')[1],text,'费率适用索引（含原文及组织说明）',e.path,e.sha256,{'kind':'rate_scope_index','original_source_id':e.id,'original_location':e.location}))
    return base+extras

def main():
    prior=json.loads((OUT/'exp32_plan.json').read_text(encoding='utf-8'))
    original=[c for c in prior['cases'] if c['group']!='transfer8']
    cases=[{**c,'round':1,'trial_group':'original127'} for c in original]
    cases += [{**c,'round':r,'trial_group':'repeat12'} for r in [2,3] for c in original if c['id'] in FOCUS]
    cases += [{'id':i,'question':q,'known':{},'group':'new12','trial_group':'new12','round':r} for r in [1,2] for i,q in NEW]
    raw=(OUT/'exp29_sources.json').read_bytes()
    base=[Evidence(**e) for e in json.loads(raw)['evidence']]
    candidate=organized_sources(base)
    from dataclasses import asdict
    evidence_data={'evidence':[asdict(e) for e in candidate]}
    plan={'cases':cases,'prompts':{'v2':fewshot_system('v2'),'v3_organized':fewshot_system('v3')},'model':cloud.SLOW_MODEL,
        'base_source_sha256':hashlib.sha256(raw).hexdigest(),
        'candidate_sources_sha256':hashlib.sha256(json.dumps(evidence_data,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
        'design':'Fresh paired composite intervention; prompt and source organization changed together, cannot isolate either contribution. Original127 once, prior12 failures three rounds, new synthetic12 two rounds. No gold or old answers in model requests.'}
    folder=OUT/'exp33_records';folder.mkdir(exist_ok=True)
    path=OUT/'exp33_plan.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==plan,'Frozen plan changed'
    else:
        save(path,plan);save(OUT/'exp33_organized_sources.json',evidence_data)
    answerer=EvidenceAnswerer()
    def one(variant,case):
        row={**case,'variant':variant}
        try:
            row.update(answerer.answer(case['question'],case['known'],base if variant=='v2' else candidate,fewshot=True,fewshot_version='v2' if variant=='v2' else 'v3'))
        except Exception as exc:
            row['error_type']=type(exc).__name__
        return row
    def name(v,c):return f"{v}_{c['id']}_r{c['round']}.json"
    with ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(one,v,c) for c in cases for v in plan['prompts'] if not (folder/name(v,c)).exists()]
        for job in as_completed(jobs):
            row=job.result();save(folder/name(row['variant'],row),row)
            print(json.dumps({k:row.get(k) for k in ['id','round','variant','total_s','error_type']},ensure_ascii=False),flush=True)
    rows=[json.loads((folder/name(v,c)).read_text(encoding='utf-8')) for c in cases for v in plan['prompts']]
    save(OUT/'exp33_minimum_bound.json',{'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':rows})

if __name__=='__main__':main()
