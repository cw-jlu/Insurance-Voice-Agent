"""Frozen same-evidence pilot: 13 prior errors, 8 controls, 8 unseen questions.

Both prompts get fresh samples; no prior answers enter requests. No answer repair.
"""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments')]
from agent.evidence_answer import Evidence, EvidenceAnswerer, BOUND_SYSTEM, fewshot_system
from agent import cloud

OUT = ROOT/'experiments/results'
FAILS = ['C05','C09','C12','C19','C20','N07','N08','N09','E05','E08','E09','E19','E31']
CONTROLS = ['C01','C07','Y012','Y015','Y036','N10','E13','E25']
NEW = [
 ('F01','普通住院合规总费用32000元，医保报了11000元，年度免赔已抵扣6000元，没有其他补偿，以有医保身份投保且已医保结算，这次赔多少？'),
 ('F02','一般医疗和门急诊共享一万元年度免赔，今年已经抵扣2500元；这次门诊医保后个人合规费用6500元，没有其他补偿，这次基础门急诊能赔多少？'),
 ('F03','57岁，门急诊加油包去年已经买了，今年在上一张保单期满后指定期限内重新投保，可以按57岁费率算吗？不是第一次买。'),
 ('F04','我投保前已有囊肿，投保以后超过等待期为这个囊肿住院手术，另购住院补偿包是不是就能赔20%？'),
 ('F05','我投保前有囊肿，后来车祸骨折住院，与囊肿无关。会不会因为有囊肿就把骨折也按既往症拒赔？'),
 ('F06','我34岁女性，重疾关爱包每5万元保额的年缴标准保费多少？不要介绍给付和等待期，也不要加基础保费。'),
 ('F07','基础住院普通部的床位费不限额吗？你只有保障摘要，缺完整合同的话请说明能确认到哪。'),
 ('F08','我39岁、有医保，做危险货物运输作业，年龄符合是不是就可以买？'),
]

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')

def main():
    folder = OUT/'exp30_records'
    folder.mkdir(exist_ok=True)
    old = {r['id']:r for r in json.loads((OUT/'exp29_questions.json').read_text(encoding='utf-8'))['cases']}
    cases = [{**old[i],'group':'prior_errors13'} for i in FAILS]
    cases += [{**old[i],'group':'controls8'} for i in CONTROLS]
    cases += [{'id':i,'question':q,'known':{},'group':'new8'} for i,q in NEW]
    prompts = {'control':BOUND_SYSTEM,'fewshot':fewshot_system()}
    sources = (OUT/'exp29_sources.json').read_bytes()
    plan = {'cases':cases,'prompts':prompts,'model':cloud.SLOW_MODEL,
            'source_sha256':hashlib.sha256(sources).hexdigest(),
            'design':'Same frozen exp29 evidence; 29 cases x 2 fresh calls; diagnostic pilot, not population accuracy.'}
    frozen = OUT/'exp30_plan.json'
    if frozen.exists():
        assert json.loads(frozen.read_text(encoding='utf-8')) == plan, 'Frozen plan changed'
    else:
        save(frozen,plan)
    evidence = [Evidence(**e) for e in json.loads(sources)['evidence']]
    answerer = EvidenceAnswerer()
    def one(variant,case):
        row={**case,'variant':variant}
        try:
            row.update(answerer.answer(case['question'],case['known'],evidence,
                bound=variant=='control',fewshot=variant=='fewshot'))
        except Exception as exc:
            row['error_type']=type(exc).__name__
        return row
    with ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(one,v,c) for c in cases for v in prompts
              if not (folder/f"{v}_{c['id']}.json").exists()]
        for job in as_completed(jobs):
            row=job.result()
            save(folder/f"{row['variant']}_{row['id']}.json",row)
            print(json.dumps({k:row.get(k) for k in ['variant','id','total_s','error_type']},ensure_ascii=False),flush=True)
    rows=[json.loads((folder/f"{v}_{c['id']}.json").read_text(encoding='utf-8')) for c in cases for v in prompts]
    save(OUT/'exp30_fewshot.json',{'plan_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),'rows':rows})

if __name__=='__main__':
    main()
