"""Opt-in conversation adapter for one-pass evidence answers and post handoff."""
from agent.evidence_answer import EvidenceAnswerer
from agent.pipeline import Answer, Pipeline
from agent.consultation import validate_progress, missing_questions
from agent.handoff import handoff_request


class EvidencePipeline(Pipeline):
    post_answer_routing = True

    def __init__(self, evidence, client=None, answerer=None, consultation_tracking=True):
        super().__init__(None,client=client,direct_answer=True)
        self.evidence=list(evidence)
        self.answerer=answerer or EvidenceAnswerer(client=self.client)
        self.consultation_tracking = consultation_tracking

    def retrieve(self,question,k=4):
        return []  # The answer receives the complete evidence pack.

    def answer(self,question,known=None,voice=False,agent_steps=0,agent_budget_s=3.0,
               understanding=None,on_sentence=None):
        # Flow understanding may update the profile, but never pre-routes gaps or
        # rewrites this question. Evidence and original words go to one generation.
        shown = dict(known or {})
        user_texts = shown.pop('_user_texts', [])
        condition_texts = shown.pop('_condition_user_texts',user_texts)
        result=self.answerer.answer(question,shown,self.evidence,allow_handoff=True,
                                   handoff_version='v4' if self.consultation_tracking else 'v2')
        if not result['outcome_validation']['valid'] or not result['provenance']['valid']:
            return Answer('本次回答暂时未能完成，请稍后再试。',route='证据-响应异常',
                outcome={'action':'error'},fallback_used=True,
                timings={'total':result['total_s'],'evidence_result':result})
        outcome=result['outcome']
        progress = None
        text = result['text']
        if self.consultation_tracking:
            try:
                cited_ids=list(dict.fromkeys(i for s in result['structured'].get('segments',[]) for i in s.get('source_ids',[])))
                progress = validate_progress(result['structured'].get('consultation'),shown['_consultation'],
                                             user_texts,shown,question,condition_texts,cited_ids)
            except (ValueError,KeyError,TypeError) as exc:
                return Answer('本次咨询暂未完成，请稍后再试。',route='证据-待办异常',
                    outcome={'action':'error'},fallback_used=True,
                    timings={'total':result['total_s'],'evidence_result':result,'consultation_fault':str(exc)})
            tasks={t['id']:t for t in progress['tasks']}
            previous_answered = {t['id'] for t in shown['_consultation'].get('tasks',[]) if t['status']=='answered'}
            segments=result['structured'].get('segments',[])
            delivered=set()
            try:
                for segment in segments:
                    ids=segment.get('task_ids',[])
                    if not isinstance(ids,list) or not all(isinstance(i,str) and i in tasks for i in ids):
                        raise ValueError('Invalid spoken task ids')
                    delivered.update(ids)
                seeded={t['id'] for t in shown['_consultation']['tasks'] if t.get('seeded')}
                unpublished={i for i in seeded if tasks[i]['status']=='answered' and i not in previous_answered and i not in delivered}
                if unpublished:raise ValueError('Seeded answered tasks lack spoken segments')
            except ValueError as exc:
                return Answer('本次咨询暂未完成，请稍后再试。',route='证据-待办异常',outcome={'action':'error'},fallback_used=True,
                    timings={'total':result['total_s'],'evidence_result':result,'consultation_fault':str(exc)})
            needs_user = missing_questions(progress)
            monetary_missing = any(t['kind'] in ('price','price_difference','claim_calculation') and
                                   t['status']=='needs_user' and t['missing_conditions'] and
                                   not t.get('is_example') for t in progress['tasks'])
            if needs_user and (progress['guard_faults'] or outcome['action']=='answer' or monetary_missing):
                # Block an ungrounded business answer before any sentence is spoken.
                safe=[s for s in segments if s.get('task_ids') and all(
                    tasks[i]['status'] in ('answered','needs_review') for i in s['task_ids'])]
                released={i for s in safe for i in s['task_ids']}
                text = ''.join(s['text'] for s in safe)+'要回答尚未完成的部分，请先确认：'+'；'.join(needs_user)+'？'
                # Any new answer suppressed along with the unsafe calculation
                # remains due; it cannot be marked delivered in the TODO ledger.
                for task in progress['tasks']:
                    if task['status']=='answered' and task['id'] not in previous_answered and task['id'] not in released:
                        task['status']='needs_answer'
                        task.pop('quoted_amount',None)
                progress['status']='pending'
                result['released_segments']=safe
                outcome = {'action':'clarify','reason_kind':'missing_user_condition',
                           'reason':'咨询依赖条件尚未确认','pending_questions':needs_user,'source_ids':[]}
            elif outcome['action']=='clarify' and not any(ch in text for ch in ('？','?')) and not any(w in text for w in ('请告诉','请提供','请补充','请确认')):
                text += '请问：'+'；'.join(needs_user or outcome['pending_questions'])+'？'
            if outcome['action']=='answer' and progress['status']!='complete':
                return Answer('本次咨询尚有未完成的问题，请稍后再试。',route='证据-待办异常',
                    outcome={'action':'error'},fallback_used=True,
                    timings={'total':result['total_s'],'evidence_result':result,'consultation':progress})
            result['effective_outcome'] = outcome
            result['effective_text'] = text
            if outcome['action']=='handoff':
                pending = list(dict.fromkeys(outcome['pending_questions']+[t['question'] for t in progress['tasks'] if t['status'] in ('needs_user','needs_review')]))
                outcome = {**outcome,'pending_questions':pending}
                result['handoff_request'] = handoff_request(question,shown,{**result,'outcome':outcome,'text':text})
                result['handoff_request']['consultation'] = progress
                progress['status'] = 'handed_off'
            else:
                result['handoff_request'] = None
            result['effective_outcome'] = outcome
        if on_sentence is not None and not self.consultation_tracking:
            on_sentence(result['text'])
        return Answer(text,route='证据-'+outcome['action'],
            understanding=understanding or {},outcome=outcome,
            handoff_request=result['handoff_request'],
            timings={'total':result['total_s'],'trace':[{'stage':'generate','calls':1},
                {'stage':'decision','outcome':outcome},{'stage':'release','text':text}],
                'evidence_result':result,**({'consultation':progress} if progress is not None else {})})
