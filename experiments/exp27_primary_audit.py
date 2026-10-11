"""Primary agent's source review of every bound_full final answer.

These are explicitly recorded review decisions, not a scoring model or regex
inference. See each row for its rationale; an independent review can disagree.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results'

# facts_correct, complete, conditions_ok, semantic citation support, review note
DECISIONS={
 'C01':(False,True,False,False,'把医院、社保投保身份及结算前提并列后说否则60%，扩大60%适用范围；该段也未引用医院范围。'),
 'C02':(True,True,False,False,'三问均有结论，但质子重离子省去仅中国大陆的必要地域限制；未将原文全部作用条件保留在口播。'),
 'C03':(True,True,True,True,'一般责任假设、免赔余额/补偿、5000/3000和自付10000/12000正确，60%投保身份前提明确。'),
 'C04':(True,True,True,True,'基础与加油包分清，1万免赔/共享300万与50%次限300年1万、年龄均正确；未问结算方式不要求枚举全部分支。'),
 'C05':(True,True,True,True,'中风不直接认定合同重疾；康复16种180日及机构80/60、护工指定审核直付500/日准确。'),
 'C06':(True,True,True,False,'5岁年龄、687/99、药省保指定互联网药店50%月100年1000和6岁含处方限制有据；最后一段未直接引用包归属元数据。'),
 'C07':(True,True,False,False,'把泛称重病套到约定重疾的转诊及护工，未说明须符合合同定义；所引原文含该限定，口播不能泛化。'),
 'C08':(True,True,True,True,'需加购、同一投保人同一订单、仅一般医疗及不含一般门急诊、三人10%有据。'),
 'C09':(True,False,False,True,'相关条款条件本身正确，但用户明确已有高血压，答案绕到如果不是既往症，也未说明无法保证核保承保，未充分回答此人当前情境。'),
 'C10':(True,True,True,True,'事故不受疾病等待限制、疾病30日与等待期确诊免责、60%有社保投保身份均明确。'),
 'C11':(True,True,True,True,'普通部与指定民营特需/国际/VIP区分、加购且约定重疾、床位1500及房型除外准确。'),
 'C12':(True,True,True,False,'正常补偿包100%、结节20%、一般医疗1万免赔及有社保身份未结算60%可作分项概述；首段引用未直接覆盖加油包归属和0免赔结构字段。'),
 'C13':(True,True,True,True,'基础重疾医疗报销和另购重疾关爱5万给付，约定100种有明确区别。'),
 'C14':(True,True,True,True,'已购前提明确，普通(10-3)*100=700、ICU3*500=1500、叠加2200正确；明确是试算。'),
 'C15':(True,True,True,True,'近视/牙科免责保留意外例外，事故和疾病等待关系准确，住院免赔与60%身份有据。'),
 'C16':(True,True,True,True,'原因免责未扩大成孕妇全部疾病免责；无关疾病按责任及既往症/等待限制，60%身份前提明确。'),
 'C17':(True,True,True,True,'问医保后剩余赔付比例，按净个人负担语境理解；基础/重疾免赔和特药目录内剩余100%，60%身份前提均有据，不要求再列完整购药申请流程。'),
 'C18':(True,True,True,True,'按58岁年龄资格回答包选择，门诊55与其它70正确；这不是个案核保承诺。'),
 'C19':(True,False,False,False,'合同段一万/指定机构与非指定60有出处，但未区分D52服务路径不限额、D53/D54资质路径，也未说明源内限额冲突，不能覆盖所问申请和额度。'),
 'C20':(False,True,False,False,'已有结节作为既往症除外后，仍说加包该责任对结节仍可赔20%，再退回是否认定既往症，容易误导当前已有结节可以加包赔。'),
 'C21':(True,True,True,True,'住院前后30日及外购药合同特定条件、医院、免赔和社保身份准确。'),
 'C22':(True,True,True,True,'康复救护车1000/16种180日与一般门急诊救护车分别解释，未一律否定所有转院救护车。'),
 'C23':(True,True,True,True,'重度癌症初诊30日、两家日本医院、床位1500及房型除外，费用种类排除有据。'),
 'C24':(True,True,True,True,'癌症重度/13种罕见病、指定药店处方审核、同项共享600万和社保目录比例正确。'),
 'C25':(True,True,True,True,'未保证货车司机职业合格，636年/63月正确；仍可更充分利用已提供特殊职业附录说明营运4吨以上等限制并追问吨位。'),
 'C26':(True,True,True,True,'年龄30明确选[26,30]，318/31、31*12=372、差54及来源行准确。'),
 'C27':(True,True,True,True,'加购共享范围和四人15%正确，社保身份改变保费不否定家庭共享；未把同一人的责任共享误判为家庭共享。'),
 'C28':(True,True,True,True,'抑郁症精神行为障碍及既往症免责，无承保保证；配置身份支持能力边界。'),
 'C29':(True,False,False,False,'对肿瘤仅给一般医疗扣1万，缺少若符合合同重疾可按重疾责任无免赔的关键分支；不是所有肿瘤费用都按一般医疗处理。'),
 'C30':(True,True,True,False,'合规、免赔未用、无补偿假设正确，共扣一次赔15000/9000，社保身份明确；第一段把所有急诊归入一般医疗却未引R02，属于跨责任引用不全。'),
 'Y001':(True,True,True,False,'基础门急诊和可选门诊包/ICU津贴分清，所有金额及身份正确；加油包归属两段仅引用描述单元格，缺直接归属元数据。'),
 'Y002':(True,True,True,True,'一般共享1万、疾病30日与事故条件、重疾无免赔有据；不因未逐一重复意外例外而判失败。'),
 'Y003':(True,True,True,True,'重疾300万0免赔、身份未结算60，护工机构审核直付及不可自主申请均有据。'),
 'Y004':(True,True,True,False,'特药审核/药店、重度初诊条件和质子大陆限制、额度比例房型准确；两处必选归属仅有条款句引用未引头部字段。'),
 'Y005':(True,True,True,True,'R16原文等待7日后因意外或疾病，未把R01的意外无等待错误移用；月100年1000及直付50%需加购准确。'),
 'Y006':(True,True,True,True,'首次门诊包55、次限300年1万、50/30的社保身份条件正确。'),
 'Y007':(True,True,True,True,'事故无等待、疾病30日、等待期内确诊医疗费免责有据。'),
 'Y008':(True,True,True,False,'普通100次免赔3、ICU500次免赔0、需另购和叠加正确；最后段只有R14头和R15叠加句，未直接引ICU加包归属。'),
 'Y009':(True,True,True,True,'限定特定疾病康复责任，零免赔及机构80/60、身份未结算48/36有据，不要求比例短问穷尽180日期限。'),
 'Y010':(True,True,True,True,'优惠人数及关系正确，家庭加购共享与单人责任共享分清，未把门急诊排除扩大到个人共享。'),
 'Y011':(True,True,True,True,'另购明确，普通/ICU日额、免赔、30/60及叠加正确。'),
 'Y012':(True,True,True,True,'另购明确，ICU500、0次免赔、30/60与叠加有据。'),
 'Y013':(True,True,True,True,'约定重疾、机构服务及审核直付不可自主理赔有据，不承诺自聘护工领现金。'),
 'Y014':(True,True,True,True,'500标准间、约定重疾和跨省转院证明、年1万100%有据。'),
 'Y015':(False,True,False,False,'1500及特需加包/重疾/房型正确，但只在该包有约定与无该包就不在整份计划范围的绝对排他结论无足够证据；R05另有床位限额，不能据缺少特需措辞作全计划排除。'),
 'Y016':(True,True,True,True,'需另购、年1万次300、50/30社保身份准确。'),
 'Y017':(True,True,True,True,'39无社保年1330，夫妻同单每人95%为1263.5；月131*95%=124.45，对应原始闭区间及折扣。'),
 'Y018':(True,True,True,True,'6岁基础279加药省保99=378，未错误补充6岁可以开处方药。'),
 'Y019':(True,True,True,False,'30岁318+141=459正确；合计所在句只直接引用141加油包费率，未给基费引用，计算输入需在同句绑定。'),
 'Y020':(True,True,True,True,'四人15及同一投保人/订单、首次或指定路径重新投保、优惠不叠加条件准确。'),
 'Y021':(True,True,True,True,'在用户有社保语境内明确投保身份及结算条件，1万免赔100/60与个人支付试算有据；不将其理解成对无社保投保的一般性排除。'),
 'Y022':(True,True,True,True,'清单范围、社保后剩余100%及有社保身份未结算60%、0免赔600万正确；此处问比例不要求完整申请流程。'),
 'Y023':(True,True,True,False,'将60%明确条件化而非对所有未走医保者下结论，并指出无社保身份不应强套60%；无社保具体比例留白可改进，但不计错误承诺；无社保另确认句仅靠例外原文不能证明需另确认。'),
 'Y024':(True,True,True,True,'ICU住院医疗与津贴区分，60%身份在同一回答上一句已明确，可承接语境，不要求每次摘要再重复；并非错误泛化所有无社保身份。'),
 'Y025':(True,True,True,True,'基础共享1万、重疾0及门诊包0次免赔300准确；未声称加包自带基础。'),
 'Y026':(True,True,True,False,'首次上限70与未知最高续保年龄分开，未按费率105承诺续保；未知句无清单范围引用，provenance机械检查也指出空引用。'),
 'Y027':(True,True,True,True,'非基础、需另购，普通和ICU日额/免赔/限天叠加准确。'),
 'Y028':(True,True,True,True,'5万属于重疾关爱加包及约定100种初诊/事故或等待条件正确，没有扩写未问价格。'),
 'Y029':(True,True,True,True,'34有社保月42及闭区间正确，额外家庭优惠以条件化表达。'),
 'Y030':(True,True,True,True,'49无社保月294及标准费率条件、家庭优惠规则正确。'),
 'Y031':(True,True,True,True,'49有社保年984及家庭人数折扣正确。'),
 'Y032':(True,True,True,True,'64无社保月881、年8898正确；年龄范围内可以投保按其年龄资格语境审，不作核保一定承保解读，实际话术仍应避免这种歧义。'),
 'Y033':(True,True,True,True,'69有社保年3316、首次及个人单标准、条件化家庭折扣有据。'),
 'Y034':(True,True,True,True,'69有社保年3316，首次/身份与家庭优惠限定有据。'),
 'Y035':(True,True,True,True,'激光祛斑美容免责有据，不涉及意外牙科等例外。'),
 'Y036':(True,True,True,True,'牙科矫正免责且保留意外例外。'),
 'Y037':(True,True,True,True,'醉酒事故导致医疗费用免责，不用意外无等待去消除醉酒免责。'),
 'Y038':(True,True,True,True,'无法确认慢性胃炎个案核保，既往症相关费除外及首次年龄疾病等待有据。'),
 'Y039':(True,True,True,True,'乳腺手术不保证承保，健康告知/核保与既往症费用区分，家庭关系和年龄有据。'),
 'Y040':(True,True,True,False,'一年期、指定期限重新投保费率与无理赔优选优惠有据，不承诺续保；一年保险期引用不足以单独证明所有资料缺少直接续保规则。'),
 'Y041':(True,True,True,True,'系统身份runtime来源证明AI/TTS及能力边界，未用保险合同证明AI身份。'),
}

def main():
    path=OUT/'exp27_bound_full.json'
    result=json.loads(path.read_text(encoding='utf-8'))
    assert len(result['rows'])==71 and set(r['id']for r in result['rows'])==set(DECISIONS)
    rows=[]
    for r in sorted(result['rows'],key=lambda r:r['id']):
        f,c,k,t,reason=DECISIONS[r['id']]
        rows.append({'id':r['id'],'facts_correct':f,'complete':c,'conditions_ok':k,
                     'overall':f and c and k,'traceable':t,'reason':reason,
                     'source_refs':list(dict.fromkeys(sid for seg in r['structured']['segments'] for sid in seg.get('source_ids',[])))})
    audit={'reviewer':'primary agent; source-based, not an API judge',
           'reviewed_output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
           'rubric':'fact correctness AND asked-item completeness AND necessary conditions; traceability separately requires each spoken sentence supported by its assigned sources. Conditions may be stated once and inherited in the same answer, absent explicit contradictory expansion. No forced recitation of all unasked terms.',
           'rows':rows}
    (OUT/'exp27_bound_primary_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print({key:sum(r[key]for r in rows)for key in ['facts_correct','complete','conditions_ok','overall','traceable']})

if __name__=='__main__':main()
