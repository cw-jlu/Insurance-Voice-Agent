"""Recorded primary source review of 16 new diagnostic questions per version."""
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent/'results'
NOTES={
'N01':'30岁318/31，全年372差54，闭区间及双源引用正确。',
'N02':'31岁423/42，全年504差81，邻接年龄档正确。',
'N03':'无社保投保不强套有医保身份未结算60%，一般医疗免赔后100；额外责任按其责任名理解。',
'N04':'已知医保身份未结算、无补偿、免赔未用，15000扣10000乘60%=3000正确。',
'N05':'孕妇骨折意外与分娩费用按原因区分，不因孕妇身份排除无关意外。',
'N06':'家庭共享仅一般医疗，普通门急诊不纳入，不能将个人共享误当家庭共享。',
'N07':'明确投保前已有结节；应回答这笔既往症相关费不能通过20%加包绕过免责。',
'N08':'已明确营运4吨，职业附录含4吨边界，不能留成可能不可以买或只说无法确认。',
'N09':'56岁指定期限重新投保不同于首次上限55，标准1019及月100.4正确，不承诺个人必定续投。',
'N10':'明确主条款1万与服务不限额的冲突，给100直结/60一万/机构不合格不赔路径。',
'N11':'普通700加ICU1500=2200，已购及医院等条件完整；前句不一定指实际资格而非算术错误。',
'N12':'车祸意外不受疾病30天等待，ICU两天500*2=1000，普通2天未超3天为0。',
'N13':'明确满足合同重疾定义，重疾医疗不扣1万、300万、医保身份未结算60，不能移用一般免赔。',
'N14':'自找护工发票不能自主理赔，指定/认可机构提供并直结500/30。',
'N15':'64无社保月881正确，费率105不能推出保证续保，未知断言需范围引用。',
'N16':'正好6岁属于含6不能开处方；另购、认可互联网药店、50%月100年1000/配送自付正确。',
}

def main():
    p=OUT/'exp28_transfer.json';data=json.loads(p.read_text(encoding='utf-8'))
    audit={'reviewer':'primary agent; each answer and source checked; no scoring API',
           'output_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
           'note':'Diagnostic phrasings designed after development errors, not an independent held-out benchmark.',
           'rows':[]}
    for r in sorted(data['rows'],key=lambda r:(r['variant'],r['id'])):
        overall=r['id'] not in ['N07','N08']
        bad_trace=['N09','N11','N15','N16'] if r['variant']=='previous_bound' else ['N07','N08','N09','N15']
        reason=NOTES[r['id']]
        if r['id']=='N07':
            reason+=' previous明确若既往不赔但未直接落到已知投前情境；grouped还追加未医保60%，进一步误导结节20%的适用。'
        if r['id']=='N08':reason+='两版都没直接给已知职业不符合投保条件的结论；grouped还建议尝试投保。'
        if r['id']=='N09':reason+=' 最终核保结论句只引用优惠费率，不支持核保主张。'
        if r['variant']=='previous_bound' and r['id']=='N11':reason+=' 合计句未绑定原始100/500数值单元格，虽前句有金额，严格按句引用仍不足。'
        if r['variant']=='previous_bound' and r['id']=='N16':reason+=' 所引药品赔付单元格/须知76–78未绑定父级药省保加油包归属与另购依据。'
        audit['rows'].append({'id':r['id'],'variant':r['variant'],'overall':overall,
                              'traceable':r['id'] not in bad_trace,'reason':reason})
    (OUT/'exp28_transfer_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    for v in ['previous_bound','grouped_only']:
        rr=[r for r in audit['rows']if r['variant']==v]
        print(v,'content',sum(r['overall']for r in rr),'trace',sum(r['traceable']for r in rr),
              'joint',sum(r['overall']and r['traceable']for r in rr))

if __name__=='__main__':main()
