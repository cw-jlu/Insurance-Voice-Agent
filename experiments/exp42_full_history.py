"""Fresh full-history validation; provider errors injected only in the named probe.

Real current Qwen flow/DS answers. No audio or phone transfer. All cases saved
once, with raw answers, known history and actual API usage for primary review.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'experiments')]
from agent import cloud
from agent.evidence_answer import Evidence, EvidenceAnswerer
from agent.evidence_pipeline import EvidencePipeline
from agent.session import CallSession

OUT = ROOT/'experiments/results/exp42_records'
CLIENT_FACTORY = cloud.llm_client


class RecentControl(CallSession):
    def _history_text(self, n=6):
        return super()._history_text(n=n)


class InjectedOverflow(Exception):
    status_code = 400
    body = {'error':{'code':'context_length_exceeded'}}


class Recorder:
    def __init__(self, inner=None, trace=None, inject=None):
        self.inner = inner or CLIENT_FACTORY()
        self.trace = trace if trace is not None else []
        self.inject = inject if inject is not None else [False]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.embeddings = self.inner.embeddings

    def with_options(self, **kwargs):
        return Recorder(self.inner.with_options(**kwargs),self.trace,self.inject)

    def create(self, **kwargs):
        row = {'model':kwargs['model'],'prompt_chars':sum(len(m.get('content') or '') for m in kwargs['messages'])}
        user = kwargs['messages'][-1].get('content') or ''
        if kwargs['model'] == cloud.SLOW_MODEL and '\n已知用户条件：' in user:
            row['known'] = json.loads(user.split('\n已知用户条件：',1)[1].split('\n【全部本轮证据】',1)[0])
        if self.inject[0] and kwargs['model'] == cloud.SLOW_MODEL:
            self.inject[0] = False
            row['injected_context_overflow_before_network'] = True
            self.trace.append(row)
            raise InjectedOverflow('Forced context overflow for recovery test, not a provider-capacity measurement')
        t0 = time.perf_counter()
        try:
            result = self.inner.chat.completions.create(**kwargs)
            row['seconds'] = time.perf_counter()-t0
            row['usage'] = result.usage.model_dump() if getattr(result,'usage',None) else None
            self.trace.append(row)
            return result
        except Exception as exc:
            row.update(seconds=time.perf_counter()-t0,error_type=type(exc).__name__)
            self.trace.append(row)
            raise


def evidence():
    data = json.loads((ROOT/'experiments/results/exp33_organized_sources.json').read_text(encoding='utf-8'))
    return [Evidence(**e) for e in data['evidence']]


def save(name, row):
    path = OUT/(name+'.json')
    path.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def run_turn(session, question):
    before = list(session.history)
    t0 = time.perf_counter()
    result = session.turn(question)
    return {'user':question,'history_before':before,'text':result.text,
            'profile':dict(session.profile),'history_after':list(session.history),
            'events':result.events,'timings':result.timings,'ended':session.ended,
            'handoff_request':session.handoff_request,'wall_s':time.perf_counter()-t0}


def recall_cases():
    original = json.loads((ROOT/'experiments/results/exp38_conversation.json').read_text(encoding='utf-8'))
    base = original['turns'][5]
    history = base['history_after']
    questions = [
        ('recent_price','按刚才最新的年龄和医保，月缴交十二个月比年缴多多少钱？'),
        ('early_expense','回到前面两万元住院费用的那个例子，医保报销后还剩多少合规费用？假设年度免赔一万元还完全没用，且其他责任条件都满足，没有其他补偿，保险能测算赔多少？不要让我重说前面已经给过的金额。'),
        ('early_topics','回到我前面问癌症的那三项保障，按当时的三项回答就诊医院或者购药渠道要求，不要换成别的三项。资料不能确认的部分明确说明。')]
    for repeat in range(1,3):
        variants = [('recent6',RecentControl),('full',CallSession)] if repeat == 1 else [('full',CallSession),('recent6',RecentControl)]
        for cid,question in questions:
            for variant,cls in variants:
                name=f'recall_{cid}_{variant}_r{repeat}'
                if (OUT/(name+'.json')).exists():continue
                recorder=Recorder()
                session=cls(EvidencePipeline(evidence(),client=recorder,consultation_tracking=False),voice=False,
                            history=[tuple(h) for h in history],profile=dict(base['profile']))
                session.flags.update(quoted=True,limits=True,disclosure=True)
                row={'id':cid,'variant':variant,'repeat':repeat,'scope':'real text continuation on identical saved 13-message history'}
                try:row['turn']=run_turn(session,question)
                except Exception as exc:row['error_type']=type(exc).__name__
                row['api_calls']=recorder.trace
                save(name,row)
                print(json.dumps({'saved':name,'error':row.get('error_type')},ensure_ascii=False),flush=True)


FIELDS={
    'age':r'多大|几岁|年龄|周岁|岁数',
    'has_social':r'医保|社保|医疗保险|公费医疗',
    'family_count':r'几人|几位|人数|多少人|几口|几个人|几名',
    'qualified_family':r'同一投保人|家庭成员|家庭单|订单|已有保单|家人|几人|人数',
    'remaining_deductible':r'免赔.*(?:剩|使用|用过|抵扣|扣过|已用)|(?:剩|使用|抵扣).{0,15}免赔',
    'relation':r'给谁|自己|本人|家人|被保人',
    'plan':r'计划|基础版|必选|加油包',
    'payment':r'年缴|月缴|按年|按月|缴费',
    'goal':r'投保还是|能否投保|能不能买|购买还是|理赔还是|投保.*理赔|买.*赔',
}


def relevant_replies(case, text, used, allow_correction):
    indices=[]
    for i,r in enumerate(case['user_reply_rules']):
        if i in used:continue
        if r['mode']=='after_answer':
            if allow_correction:return [i]
            continue
        if any(re.search(FIELDS.get(field,r'(?!)'),text) for field in r['requested_fields']):
            indices.append(i)
    return indices


def dialogue_cases():
    data=json.loads((ROOT/'experiments/data/complex_dialogue_v1.json').read_text(encoding='utf-8'))
    for case in data['cases']:
        name='dialogue_'+case['id']
        if (OUT/(name+'.json')).exists():continue
        recorder=Recorder()
        session=CallSession(EvidencePipeline(evidence(),client=recorder,consultation_tracking=False),voice=False)
        row={'id':case['id'],'type':case['type'],'split':case['split'],
             'scope':'real current text CallSession from opening; rule-based external operator response matching; requires primary audit',
             'opening':session.opening(),'turns':[],'operator_actions':[]}
        used=set()
        question=case['initial_user']
        try:
            for _ in range(12):
                turn=run_turn(session,question)
                row['turns'].append(turn)
                if session.ended:
                    row['stop_reason']='session_ended'
                    break
                detail=turn['timings'].get('answer_detail',{}).get('evidence_result',{})
                outcome=detail.get('outcome',{})
                allow_correction=outcome.get('action')=='answer' or ('报价' in turn['events'])
                question_sentences=[sentence for sentence in re.split(r'(?<=[。！？!?；;])',turn['text'])
                                    if '?' in sentence or '？' in sentence]
                search='\n'.join(question_sentences)+'\n'+'\n'.join(outcome.get('pending_questions',[]))
                # Only disclose in response to a real question or structured clarify.
                requested=outcome.get('action')=='clarify' or '?' in turn['text'] or '？' in turn['text']
                selected=relevant_replies(case,search,used,allow_correction)
                selected=[i for i in selected if requested or case['user_reply_rules'][i]['mode']=='after_answer']
                if not selected:
                    row['stop_reason']='no_relevant_question_or_all_conditions_disclosed'
                    break
                used.update(selected)
                question=''.join(case['user_reply_rules'][i]['utterance'] for i in selected)
                row['operator_actions'].append({'reply_indices':selected,'utterance':question,
                                               'triggering_outcome':outcome,'triggering_text':turn['text']})
            else:row['stop_reason']='diagnostic_round_cap'
        except Exception as exc:row['error_type']=type(exc).__name__
        row['undisclosed_reply_indices']=[i for i in range(len(case['user_reply_rules'])) if i not in used]
        row['api_calls']=recorder.trace
        save(name,row)
        print(json.dumps({'saved':name,'turns':len(row['turns']),'stop':row.get('stop_reason'),'error':row.get('error_type')},ensure_ascii=False),flush=True)


def overflow_probe():
    name='overflow_injected_real_summary_v2'
    if (OUT/(name+'.json')).exists():return
    history=[('用户','早期一般住院总费用20000元，医保报8000元，剩余12000元合规；年度免赔还剩10000元；无其他补偿，其他责任条件满足。'),
             ('坐席','按这些明确假设，测算2000元。')]
    # Repetitive background makes an actually smaller summary possible; this is
    # a fault-injection test, not a claim that this short prompt exceeds 1M.
    for i in range(8):
        history += [('用户',f'第{i+1}次确认：这个费用例子是之前已经说明的假设，暂时没有新增费用，不修改任何金额。'),
                    ('坐席','明白，保留原有费用及免赔前提。')]
    recorder=Recorder(inject=[True])
    engine=EvidenceAnswerer(client=recorder)
    session=CallSession(EvidencePipeline(evidence(),client=recorder,answerer=engine,consultation_tracking=False),voice=False,
                        profile={'relation':'本人','age':35,'has_social':True},history=list(history))
    session.flags.update(quoted=True,limits=True,disclosure=True)
    row={'id':name,'scope':'Injected 400 context overflow before first DS request; actual Qwen summary and actual DS retry. No natural provider overflow or production latency claimed.'}
    try:
        row['turn']=run_turn(session,'回到最早的住院费用例子，按已明确的免赔和其他假设，测算能赔多少？')
        row['original_history_unchanged']=session.history[:len(history)]==history
        row['early_original_still_retained']='20000' in session._history_text()
    except Exception as exc:row['error_type']=type(exc).__name__
    row['api_calls']=recorder.trace
    save(name,row)
    print(json.dumps({'saved':name,'error':row.get('error_type')},ensure_ascii=False),flush=True)


def main():
    global OUT, CLIENT_FACTORY
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['recall','dialogue','overflow','all'],default='all')
    parser.add_argument('--windows-https',action='store_true',help='Test-only Windows certificate-checked HTTPS; no production networking change')
    args=parser.parse_args()
    if args.windows_https:
        from windows_https_transport import native_client
        CLIENT_FACTORY=native_client
        OUT=ROOT/'experiments/results/exp42_windows_https_records'
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'date':'2026-10-13','models':{'answer':cloud.SLOW_MODEL,'flow':cloud.FAST_MODEL},
          'policy':'full heard history; overflow-only summary; no truncation by turn count',
          'no_audio':True,'operator_rules_not_sent_to_agent':True,
          'transport':'Windows HTTPS test-only; Python TLS failed on this host' if args.windows_https else 'production Python SDK',
          'code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
              ['src/agent/context.py','src/agent/cloud.py','src/agent/session.py','src/agent/evidence_answer.py','src/agent/pipeline.py','src/agent/small_model.py']}}
    plan_name = ('exp42_windows_https_' if args.windows_https else 'exp42_')+args.mode+'_plan.json'
    (OUT.parent/plan_name).write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.mode in ('recall','all'):recall_cases()
    if args.mode in ('dialogue','all'):dialogue_cases()
    if args.mode in ('overflow','all'):overflow_probe()


if __name__=='__main__':main()
