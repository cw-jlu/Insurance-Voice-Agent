"""Frozen paired one-pass answers vs post-answer handoff, with independent policy labels.

Labels are NEVER sent to the answering model. Every decision and any partial
spoken facts are manually audited; handoff does not erase a false assertion.
"""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from agent.evidence_answer import Evidence, EvidenceAnswerer, fewshot_system, handoff_system
from agent import cloud
from exp30_fewshot import save

OUT=ROOT/'experiments/results'
NEW=[
 ('H01','36岁，有社保，第一次买必选计划，年缴标准保费是多少？只报价。',['answer']),
 ('H02','我想买的阿帕替尼一定在这份特药清单里吗？请确认这个药能不能按本保单报。',['handoff']),
 ('H03','我的原位癌诊断能不能按本合同100种重大疾病无免赔报？本轮能否确认定义？',['handoff']),
 ('H04','我现在有肾囊肿，下周才买保险，以后为同一囊肿手术，即使加住院补偿包也能赔20%吗？',['answer']),
 ('H05','基础住院费用15000元，没走医保，想算能赔多少。投保时选有医保还是无医保我没说，年度免赔还剩多少也没说。',['answer','clarify']),
 ('H06','已符合合同100种重疾定义的化疗和获批指定药店院外特药，是不是都要扣1万元？只比较免赔。',['answer']),
 ('H07','已确定满足合同16种特定疾病康复责任的全部条件，康复住院有没有免赔额？',['answer']),
 ('H08','癌症基因检测原文主体限1万，增值服务却写不限额。我准备通过增值服务申请，你能确定我这次绝对不限额吗？',['handoff']),
 ('H09','我40岁，有社保，是办公室文员，已经做过乳腺肿瘤手术，核保一定能让我买这份保险吗？请判断我的承保结果。',['handoff']),
 ('H10','第一次买必选计划，年缴多少钱？我的年龄和社保情况还没告诉你。',['clarify']),
 ('H11','我69岁有社保，首次买必选计划，年缴标准保费多少？不要讲续保。',['answer']),
 ('H12','我69岁有社保，在上一张保单期满后的指定期限内重新投保，能确认年缴仍然3316元吗？',['handoff']),
 ('H13','只问本轮是否有完整民营医院名单和100种重疾逐项定义附件，不要给具体医院和病名。',['answer']),
 ('H14','基因检测服务备注写不限额，是不是就能说所有基因检测都不限额？只回答能不能这么概括。',['answer']),
 ('H15','我30岁有社保，第一次买必选计划，年缴多少钱？另外原位癌能否按这份合同的100种重疾报销，请确认定义。',['handoff']),
 ('H16','已购药省保，指定平台合规药费100元，已过7天等待期，本月已赔90元，年度还充足，这次最多赔多少？',['answer']),
]
REQUIRED={'C19','Y038','Y039','E25','E32','E33','E34'}
OPTIONAL={'C05','C07','C29','Y026','Y040'}
USER_MISSING={'C03','C11','C12','C14','C25','Y017','Y020','Y023','Y032'}

def main():
    previous=json.loads((OUT/'exp33_plan.json').read_text(encoding='utf-8'))
    cases=[{k:c[k] for k in ['id','question','known','group']} for c in previous['cases'] if c['trial_group']=='original127']
    policies=[]
    for c in cases:
        i=c['id'];allowed=['handoff'] if i in REQUIRED else ['answer','handoff'] if i in OPTIONAL else ['answer','clarify'] if i in USER_MISSING else ['answer']
        policies.append({'id':i,'allowed_actions':allowed,'required_external':i in REQUIRED,
            'reason':'需要具体附件/冲突适用或个案核保，不能只给确定承诺' if i in REQUIRED else '可依据现有原文或完整条件分支回答；缺用户条件才追问；无需因未问附件一律转人工'})
    for i,q,allowed in NEW:
        cases.append({'id':i,'question':q,'known':{},'group':'new_handoff16'})
        policies.append({'id':i,'allowed_actions':allowed,'required_external':allowed==['handoff'],
            'reason':'调用前冻结的新问题决策边界，人工核实与缺用户条件区分'})
    source_raw=(OUT/'exp33_organized_sources.json').read_bytes()
    evidence=[Evidence(**e) for e in json.loads(source_raw)['evidence']]
    plan={'cases':cases,'policies_not_sent_to_model':policies,
        'prompts':{'control':fewshot_system('v3'),'post_handoff':handoff_system()},
        'model':cloud.SLOW_MODEL,'source_sha256':hashlib.sha256(source_raw).hexdigest(),
        'design':'143 questions x2 fresh one-pass calls, identical organized evidence. Original127 separate from new16. Primary-agent policy and source audit; no independent second reviewer. No policy labels, prior answers or scores in model requests.'}
    path=OUT/'exp34_plan.json';folder=OUT/'exp34_records';folder.mkdir(exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==plan,'Frozen plan changed'
    else:save(path,plan)
    engine=EvidenceAnswerer()
    def one(v,c):
        row={**c,'variant':v}
        try:
            row.update(engine.answer(c['question'],c['known'],evidence,
                fewshot=True,fewshot_version='v3',allow_handoff=v=='post_handoff'))
        except Exception as exc:row['error_type']=type(exc).__name__
        return row
    with ThreadPoolExecutor(4) as pool:
        jobs=[pool.submit(one,v,c) for c in cases for v in plan['prompts'] if not (folder/f"{v}_{c['id']}.json").exists()]
        for job in as_completed(jobs):
            r=job.result();save(folder/f"{r['variant']}_{r['id']}.json",r)
            print(json.dumps({k:r.get(k) for k in ['id','variant','total_s','error_type','outcome']},ensure_ascii=False),flush=True)
    rows=[json.loads((folder/f"{v}_{c['id']}.json").read_text(encoding='utf-8')) for c in cases for v in plan['prompts']]
    save(OUT/'exp34_post_handoff.json',{'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':rows})

if __name__=='__main__':main()
