"""New diagnostic phrasings, frozen before calls; not an external held-out benchmark."""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from agent import cloud
from agent.evidence_answer import EvidenceAnswerer, full_evidence, grouped_evidence, BOUND_SYSTEM
from exp28_grouped_evidence import save
OUT=ROOT/'experiments/results'
QUESTIONS=[
 ('N01','我30岁有医保，基础保障按月交一年比按年交贵多少？'),
 ('N02','我31岁有医保，基础保障按月交一年比按年交贵多少？'),
 ('N03','我投保时选的是无社保身份，住院也没走医保，是不是只能报60%？'),
 ('N04','投保选了有医保，住院没有用医保结算，符合范围的费用15000元，没有其他补偿，年免赔还没用，赔多少？'),
 ('N05','我怀孕了，因摔断手臂住院的费用能报吗？生孩子的住院费也一起报吗？'),
 ('N06','三个人买了家庭共享免赔额包，我看普通门诊自己花的钱，能计入家人的共享免赔吗？'),
 ('N07','买保险之前已经查出甲状腺结节，加上住院补偿包，之后结节手术能按20%报吗？'),
 ('N08','我开营运货车，总质量4吨，42岁有医保，可以买吗？'),
 ('N09','我56岁，去年买了门急诊加油包，到期按保险公司指定期限重新投保，今年还能按这个包的费率买吗？'),
 ('N10','基因检测主条款写一万元，服务备注写不限额，是不是所有检测都不限额？'),
 ('N11','普通住院10天，其中ICU3天，已买住院津贴包，两项可以加起来拿2200元吗？'),
 ('N12','今天生效的住院津贴包，明天车祸住ICU两天，是否要等30天才有津贴？'),
 ('N13','确诊的癌症满足合同重疾定义，住院和化疗都走一般医疗扣一万元吗？'),
 ('N14','请我自己找的护工照顾重疾住院，一天500元，拿发票就能报吗？'),
 ('N15','64岁无社保，基础保障月缴881元，是不是保证我可以一直续到105岁？'),
 ('N16','孩子正好6岁，药省保能开处方药吗？网上普通购药怎么赔？'),
]

def main():
    path=OUT/'exp28_transfer.json'
    if path.exists():raise SystemExit('Frozen transfer trial already exists.')
    save(OUT/'exp28_transfer_questions.json',{'questions':QUESTIONS,
         'note':'New phrasings, designed after development errors; not independent population sampling.'})
    result={'model':cloud.SLOW_MODEL,'prompt':BOUND_SYSTEM,'rows':[]}
    packs={'previous_bound':full_evidence(labelled=True),'grouped_only':grouped_evidence()}
    answerer=EvidenceAnswerer()
    def one(qid,q,v):
        r={'id':qid,'question':q,'variant':v}
        try:r.update(answerer.answer(q,{},packs[v],bound=True))
        except Exception as exc:r['error_type']=type(exc).__name__
        return r
    with ThreadPoolExecutor(4)as pool:
        jobs=[pool.submit(one,qid,q,v)for qid,q in QUESTIONS for v in packs]
        for f in as_completed(jobs):
            r=f.result();result['rows'].append(r);save(path,result)
            print(json.dumps({k:r.get(k)for k in ['id','variant','total_s','error_type']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
