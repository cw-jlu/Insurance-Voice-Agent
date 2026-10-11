"""Expanded frozen evaluation of the located full-evidence answerer. No repairs."""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from agent import cloud
from agent.evidence_answer import EvidenceAnswerer, grouped_evidence, BOUND_SYSTEM
from exp18_voice_prompt_ablation import rows_for
from exp28_transfer import QUESTIONS
from exp28_grouped_evidence import save

OUT=ROOT/'experiments/results'
NEW=[
('E01','price','孩子4岁有医保，只买基础版，按当前标准费率年缴多少钱？'),
('E02','price','孩子正好5岁，有医保，只买基础版年缴多少钱？'),
('E03','price','55岁没有社保，基础版每月标准保费是多少？'),
('E04','price','56岁有社保，基础版年缴标准保费是多少？'),
('E05','price','31岁女性，重疾关爱加油包每5万元保额年缴多少钱？不要算基础版。'),
('E06','price','31岁男性，重疾关爱加油包每5万元保额年缴多少钱？不要算基础版。'),
('E07','calculation','普通住院合规费用8000元，已用医保结算，没其他补偿，一般医疗年免赔还没用，能赔多少？'),
('E08','calculation','普通住院合规费用20000元，其中医保报了10000元，一般医疗年免赔还没用，没有其他补偿，商业险赔多少？'),
('E09','calculation','一般医疗年度免赔已抵扣6000元，这次普通住院合规费用9000元，医保身份投保且已医保结算，没有其他补偿，赔多少？'),
('E10','calculation','我按无社保身份投保，普通住院合规费用15000元，年免赔没用，也没其他补偿，赔多少？'),
('E11','calculation','已买住院津贴包，普通住院2天，符合责任条件，这次一般住院津贴能领多少？'),
('E12','calculation','已买津贴包，符合责任条件，总共住院2天，两天都在ICU，普通和ICU津贴加起来多少？'),
('E13','calculation','已买津贴包，责任条件全符合，普通住院40天，今年没有领过津贴，一次最多领多少？'),
('E14','calculation','已买津贴包，同一病因住院两次，每次10天，出院后20天又入院，其他责任条件都满足，一般住院津贴免赔扣几次、领多少？'),
('E15','calculation','已购门急诊包，有医保身份投保且医保结算，单次合规门诊费200元，无其他补偿，今年限额没用过，赔多少？'),
('E16','calculation','已购门急诊包，以有医保身份投保却没按医保结算，单次合规门诊费200元，其他条件满足，赔多少？'),
('E17','calculation','已购药省保，指定互联网药店买100元合规药品，当月已赔90元、年度限额还充足，本次还能赔多少？'),
('E18','calculation','药省保今年已赔900元，本月未赔，指定平台这次合规药费400元，其他条件满足，本次能赔多少？'),
('E19','calculation','基础一般医疗年免赔已抵扣7000元，这次普通门急诊合规费用5000元，已医保结算且无其他补偿，基础门急诊赔多少？'),
('E20','hospital','普通疾病住院，在保单约定的指定民营医院普通部，是不是因为私立就一概不赔？'),
('E21','hospital','我买了重疾特需加油包，符合合同重疾定义，去不在保单约定名单内的私立医院VIP部，这样也能报吗？'),
('E22','preexisting','我投保前有甲状腺结节，投保后因车祸骨折住院，和结节无关，保险会不会因为有结节把骨折费用也排除？'),
('E23','preexisting','乳腺疾病投保前已确诊，投保成功后为这个同一病做手术，过了30天等待期是不是就能报？'),
('E24','coverage','投保后过等待期才首次确诊结节，不属于既往症，已购住院补偿包且按医保身份结算，合规住院费10000元，其他条件满足，这个包赔多少？'),
('E25','source_conflict','住院补偿包同时写结节20%、有医保身份未结算60%，如果两条同时满足，到底取20%、60%还是乘成12%？资料明确写组合方式了吗？'),
('E26','coverage','投保60天后才首次出现腰椎间盘突出，不是既往症，住院补偿包不赔脊椎病，是不是基础一般医疗也就一定不赔？'),
('E27','coverage','已经确认符合合同16种特定疾病的康复条件，但还没确认是否符合100种重疾，能不能直接说重大疾病医疗也无免赔？'),
('E28','coverage','只是普通疾病，没满足合同重大疾病定义，跨省住院能领重大疾病异地转诊住宿补贴吗？'),
('E29','service','符合合同重疾跨省转诊，转院证明等条件都满足，我带两名亲属一起去，两人的交通住宿费能都报吗？'),
('E30','service','合同重疾跨省转诊条件满足，坐飞机商务舱，能按商务舱价格全额报交通费吗？'),
('E31','unknown','基础一般医疗普通部住院的床位费，是不是也统一每天最多1500元？原文有这个限制吗？'),
('E32','unknown','把特药保障的完整药品清单给我，资料里有没有完整清单？别猜药名。'),
('E33','unknown','原位癌是否一定在这个合同100种重大疾病定义里？你手里的资料能确认这个定义吗？'),
('E34','unknown','能不能把所有约定民营医院的完整名单列出来？资料给全了吗？'),
('E35','coverage','基础版写外购药械费用，是不是只要有医院处方，任何院外药都能报？'),
('E36','renewal','费率表有到105岁的年龄档，是不是证明我70岁首次买后就保证续保到105岁？'),
('E37','occupation','我开的货车总质量3吨，而且非营运，特殊职业表里营运4吨及以上那一条能直接套在我身上拒保吗？'),
('E38','occupation','我是危险货物运输作业员，40岁，有医保，按这份职业表能投保吗？'),
('E39','eligibility','71岁从没买过这款保险，现在第一次买，能不能直接按71岁的费率投保？'),
('E40','family','我和配偶不同投保人、不同订单，各买家庭共享免赔包，两个订单还能合在一起只算一万元免赔吗？'),
]

def main():
    path=OUT/'exp29_expanded_locations.json'
    folder=OUT/'exp29_records'
    folder.mkdir(exist_ok=True)
    cases=[{'id':r['id'],'group':'development71','question':r['question'],
            'known':dict(r.get('context') or {})}for r in rows_for('all')]
    cases += [{'id':i,'group':'prior_boundary16','question':q,'known':{}}for i,q in QUESTIONS]
    cases += [{'id':i,'group':'new40','category':cat,'question':q,'known':{}}for i,cat,q in NEW]
    frozen=OUT/'exp29_questions.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8'))['cases']!=cases:
        raise ValueError('Questions changed during frozen trial')
    if not frozen.exists():save(frozen,{'cases':cases,'note':'New40 designed after development errors; not independently sampled.'})
    evidence=grouped_evidence()
    snapshot=OUT/'exp29_sources.json'
    serialized={'evidence':[asdict(e)for e in evidence]}
    if snapshot.exists() and json.loads(snapshot.read_text(encoding='utf-8'))!=serialized:
        raise ValueError('Sources changed during frozen trial')
    if not snapshot.exists():save(snapshot,serialized)
    answerer=EvidenceAnswerer()
    def one(case):
        row=dict(case)
        try:row.update(answerer.answer(case['question'],case['known'],evidence,bound=True))
        except Exception as exc:row['error_type']=type(exc).__name__
        return row
    with ThreadPoolExecutor(4)as pool:
        jobs=[pool.submit(one,c)for c in cases if not(folder/f"{c['id']}.json").exists()]
        for f in as_completed(jobs):
            row=f.result();save(folder/f"{row['id']}.json",row)
            print(json.dumps({k:row.get(k)for k in ['id','group','total_s','error_type']},ensure_ascii=False),flush=True)
    rows=[json.loads((folder/f"{c['id']}.json").read_text(encoding='utf-8'))for c in cases]
    save(path,{'model':cloud.SLOW_MODEL,'prompt':BOUND_SYSTEM,'snapshot':snapshot.name,
               'snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
               'design':'Located full responsibility bundles; 127 questions; one recorded answer each; no recall, reflection, repairs or answer retries',
               'rows':rows})

if __name__=='__main__':main()
