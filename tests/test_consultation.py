import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agent.consultation import start,validate_progress,quote_scope_known,add_questions
from agent.evidence_answer import Evidence,EvidenceAnswerer
from agent.evidence_pipeline import EvidencePipeline
from agent.session import CallSession


def fact(name,value,quote='',state='confirmed'):
    return {'name':name,'value':value,'user_quote':quote,'state':state}


def task(i,kind='coverage',status='answered',requires=(),amount=None):
    return {'id':f'q{i}','question':{1:'年缴保费',2:'月缴保费',3:'全年差额'}.get(i,'原问题'),
            'kind':kind,'status':status,'requires':list(requires),'is_example':False,
            **({'quoted_amount':amount} if amount is not None else {})}


def output(text,tasks,facts=(),action='answer'):
    return {'segments':[{'text':text,'source_ids':['d'],'derivation':''}],'unknowns':[],
            'consultation':{'tasks':tasks,'facts':list(facts)},
            'outcome':{'action':action,'reason_kind':{'answer':'none','clarify':'missing_user_condition','handoff':'individual_review'}[action],
                       'reason':'按本轮资料和已确认条件处理',
                       'pending_questions':[] if action=='answer' else ['原问题所需条件或审核'],
                       'source_ids':[] if action!='handoff' else ['d']}}


class ConsultationTest(unittest.TestCase):
    def session(self,outputs,slots,voice=False,tracking=False):
        engine=EvidenceAnswerer(client=object())
        pipe=EvidencePipeline([Evidence('d','测试条款与费率原文','位置','path','hash')],client=object(),answerer=engine)
        pipe.understand=lambda text,known:{'intent':'报价','slots':slots.pop(0),'gaps':[],'items':[],'topics':[]}
        s=CallSession(pipe,voice=voice,fast_lane=False,playback_tracking=tracking)
        calls=[]
        def call(system,user,model):
            calls.append(user)
            return {'model':'test','output':deepcopy(outputs.pop(0)),'seconds':0}
        engine.call=call
        return s,calls

    def test_three_original_questions_survive_both_information_only_turns(self):
        waiting=[task(1,'price','needs_user',['age','has_social']),task(2,'price','needs_user',['age','has_social']),task(3,'price_difference','needs_user',['age','has_social'])]
        done=[task(1,'price',requires=['age','has_social'],amount=2563),task(2,'price',requires=['age','has_social'],amount=253),task(3,'price_difference',requires=['age','has_social'])]
        outputs=[output('请问母亲几岁？',waiting,[fact('age',None,state='missing'),fact('has_social',None,state='missing')],'clarify'),
                 output('请问母亲有医保吗？',waiting,[fact('age',62,'母亲62岁'),fact('has_social',None,state='missing')],'clarify'),
                 output('年缴2563元，月缴253元，全年差473元。',done,[fact('age',62,'母亲62岁'),fact('has_social',True,'她有居民医保')])]
        s,calls=self.session(outputs,[{'relation':'父母'},{'age':62},{'has_social':True}])
        original='给母亲首次买必选计划个人单，年缴、月缴、全年差额各多少？'
        with patch('agent.small_model.classify_act',side_effect=['提问','提供信息','提供信息']):
            s.turn(original);s.turn('母亲62岁。');r=s.turn('她有居民医保。')
        self.assertEqual(len(calls),3)
        self.assertIn(original,calls[-1])
        self.assertIn('全年差额',calls[-1])
        self.assertIn('473',r.text)
        self.assertNotIn('投保链接',r.text)
        self.assertEqual(s.consultation['status'],'complete')
        self.assertTrue(s.flags['quoted'])
        self.assertEqual(len(s.consultation['tasks']),3)

    def test_corrected_conditions_reopen_completed_difference_without_question_mark(self):
        tasks=[task(1,'price',requires=['has_social'],amount=423),task(2,'price',requires=['has_social'],amount=42),task(3,'price_difference',requires=['has_social'])]
        fixed=deepcopy(tasks);fixed[0]['quoted_amount']=971;fixed[1]['quoted_amount']=96
        outputs=[output('年423元、月42元，全年差81元。',tasks,[fact('age',35,'35岁'),fact('has_social',True,'35岁有医保')]),
                 output('年971元、月96元，全年差181元。',fixed,[fact('age',35,'35岁'),fact('has_social',False,'没有医保')])]
        s,calls=self.session(outputs,[{'age':35,'has_social':True},{'has_social':False}])
        with patch('agent.small_model.classify_act',side_effect=['提问','提供信息']):
            s.turn('我35岁有医保，首次买必选计划，年/月保费和全年差额多少？')
            r=s.turn('说错了，没有医保，三个金额重算。')
        self.assertEqual(len(calls),2)
        self.assertIn('181',r.text)
        self.assertIn('继续原咨询',r.events)
        self.assertFalse(any(f['value'] is True for f in s.consultation['facts'] if f['name']=='has_social'))

    def test_incomplete_individual_claim_is_asked_not_released_or_counted_done(self):
        raw=output('可赔3000元。',[task(1,'claim_calculation')],[])
        s,calls=self.session([raw],[{}])
        with patch('agent.small_model.classify_act',return_value='提问'):
            r=s.turn('合规住院费用15000元，没走医保，实际还能赔多少？')
        self.assertNotIn('3000',r.text)
        self.assertIn('投保时',r.text)
        self.assertIn('剩多少免赔',r.text)
        self.assertEqual(s.consultation['status'],'pending')
        self.assertEqual(r.timings['answer_detail']['evidence_result']['text'],'可赔3000元。')
        self.assertIn('咨询依赖条件尚未确认',r.timings['answer_detail']['evidence_result']['effective_outcome']['reason'])

    def test_unasked_assumption_and_wrong_source_are_not_confirmed_facts(self):
        previous=start('我这次能赔多少？','c1')
        raw={'facts':[fact('insured_with_social',True,'没走医保'),fact('remaining_deductible',10000,'今年没用过免赔')],
             'tasks':[task(1,'claim_calculation')]}
        state=validate_progress(raw,previous,['没走医保'],{},'这次能赔多少？')
        self.assertEqual(state['tasks'][0]['status'],'needs_user')
        self.assertTrue(all(f['state']=='missing' for f in state['facts']))

    def test_explicit_example_is_allowed_without_turning_assumptions_into_profile(self):
        raw=output('按这个假设测算3000元。',[{**task(1,'claim_calculation',requires=['insured_with_social','remaining_deductible']),'is_example':True}],
                   [fact('insured_with_social',True,state='assumed'),fact('remaining_deductible',10000,state='assumed')])
        s,_=self.session([raw],[{}])
        with patch('agent.small_model.classify_act',return_value='提问'):
            r=s.turn('举个例子，假设有医保身份投保但没结算，剩余免赔一万，合规费用一万五会怎么算？')
        self.assertIn('3000',r.text)
        self.assertNotIn('insured_with_social',s.profile)
        self.assertNotIn('remaining_deductible',s.profile)
        self.assertTrue(all(f['state']=='assumed' for f in s.consultation['facts']))

    def test_clarify_metadata_is_made_audible_if_model_omitted_question(self):
        raw=output('首次年龄到70岁。',[task(1,'eligibility','needs_user',['age'])],[fact('age',None,state='missing')],'clarify')
        s,_=self.session([raw],[{}])
        with patch('agent.small_model.classify_act',return_value='提问'):
            r=s.turn('父亲首次买，年龄合适吗？')
        self.assertIn('被保人现在多少周岁',r.text)
        self.assertIn('？',r.text)

    def test_silently_dropped_old_tasks_are_rejected_and_no_sales_resume(self):
        outputs=[output('请问年龄？',[task(1,'price','needs_user',['age']),task(3,'price_difference','needs_user',['age'])],[fact('age',None,state='missing')],'clarify'),
                 output('年423元。',[task(1,'price',amount=423)])]
        s,_=self.session(outputs,[{}, {'age':35}])
        with patch('agent.small_model.classify_act',side_effect=['提问','提供信息']):
            s.turn('年费和全年差额多少？');r=s.turn('35岁。')
        self.assertEqual(s.consultation['status'],'error')
        self.assertNotIn('423',r.text)
        self.assertFalse(s.flags['quoted'])

    def test_unheard_completion_is_reopened_and_late_receipt_cannot_finish_tasks(self):
        s,_=self.session([output('年423元。月42元。',[task(1,'price',amount=423),task(2,'price',amount=42)],
                                [fact('age',35,'35岁'),fact('has_social',True,'有医保')])],
                         [{'age':35,'has_social':True}],voice=True,tracking=True)
        with patch('agent.small_model.classify_act',return_value='提问'):
            r=s.turn('我35岁有医保，首次买必选个人单，年/月多少钱？')
        self.assertEqual(s.consultation['status'],'awaiting_playback')
        s.confirm_playback(r.playback_id,'年423元。')
        s.interrupt_playback(r.playback_id)
        self.assertEqual(s.consultation['status'],'pending')
        self.assertTrue(all(t['status']=='pending' for t in s.consultation['tasks']))
        self.assertTrue(all('quoted_amount' not in t for t in s.consultation['tasks']))
        self.assertFalse(s.confirm_playback(r.playback_id,r.text,True))

    def test_completed_playback_commits_task_completion(self):
        s,_=self.session([output('答案已说完。',[task(1)])],[{}],voice=True,tracking=True)
        with patch('agent.small_model.classify_act',return_value='提问'):
            r=s.turn('等待期是多少？')
        self.assertTrue(s.confirm_playback(r.playback_id,r.text,True))
        self.assertEqual(s.consultation['status'],'complete')

    def test_pure_backchannel_does_not_complete_or_advance_pending_consultation(self):
        s,calls=self.session([output('请问年龄？',[task(1,'price','needs_user',['age'])],[fact('age',None,state='missing')],'clarify')],[{},{}])
        with patch('agent.small_model.classify_act',side_effect=['提问','附和']):
            s.turn('首次必选年缴多少钱？');r=s.turn('嗯。')
        self.assertEqual(len(calls),1)
        self.assertEqual(s.consultation['status'],'pending')
        self.assertNotIn('投保链接',r.text)

    def test_explicit_handoff_preserves_all_active_questions(self):
        s,calls=self.session([output('请问年龄？',[task(1,'price','needs_user',['age']),task(3,'price_difference','needs_user',['age'])],[fact('age',None,state='missing')],'clarify')],[{},{}])
        with patch('agent.small_model.classify_act',side_effect=['提问','要人工']):
            s.turn('年/月费和全年差额多少？');s.turn('转人工客服。')
        self.assertEqual(len(calls),1)
        self.assertEqual(s.handoff_request['status'],'requested')
        self.assertEqual(len(s.handoff_request['consultation']['tasks']),2)

    def test_program_quote_rejects_unknown_or_outside_standard_scope(self):
        profile={'age':35,'has_social':True}
        self.assertTrue(quote_scope_known(profile,['自己首次买必选计划个人单']))
        for text in ['35岁有医保','续保个人单价格','首次必选个人单加住院津贴加油包','一家三口首次必选个人单']:
            self.assertFalse(quote_scope_known(profile,[text]))
        self.assertFalse(quote_scope_known({'age':35,'has_social':'无医保'},['首次必选个人单']))

    def test_correction_with_consent_cannot_bypass_original_questions_or_send_link(self):
        tasks=[task(1,'price',requires=['has_social'],amount=423),task(3,'price_difference',requires=['has_social'])]
        fixed=deepcopy(tasks);fixed[0]['quoted_amount']=971
        s,calls=self.session([output('年423元、全年差81元。',tasks,[fact('age',35,'35岁'),fact('has_social',True,'35岁有医保')]),
                              output('年971元、全年差181元。',fixed,[fact('age',35,'35岁'),fact('has_social',False,'没有医保')])],
                             [{'age':35,'has_social':True},{'has_social':False}])
        with patch('agent.small_model.classify_act',side_effect=['提问','同意投保']):
            s.turn('我35岁有医保，首次必选个人单，年费与全年差额多少？')
            s.flags.update(limits=True,disclosure=True)
            r=s.turn('说错了，没有医保，三个金额重算，行，发链接吧。')
        self.assertEqual(len(calls),2)
        self.assertIn('181',r.text)
        self.assertFalse(s.flags['link'])

    def test_handoff_request_includes_every_unfinished_original_task(self):
        t=[task(1,'coverage','needs_review'),task(2,'other','needs_review')]
        raw=output('两项需要人工核实。',t,action='handoff')
        s,_=self.session([raw],[{}])
        with patch('agent.small_model.classify_act',return_value='提问'):
            s.turn('这两个定义能确认吗？')
        self.assertEqual(s.consultation['status'],'handed_off')
        self.assertEqual(len(s.handoff_request['consultation']['tasks']),2)
        self.assertIn('年缴保费',s.handoff_request['pending_questions'])

    def test_clarify_cannot_release_unrequested_monetary_assumptions_or_drop_other_answers(self):
        tasks=[task(1,'claim_calculation','needs_user'),task(2,'coverage')]
        raw=output('假设免赔一万，可赔3000元。等待期30天。请问剩多少免赔？',tasks,action='clarify')
        s,_=self.session([raw],[{}])
        with patch('agent.small_model.classify_act',return_value='提问'):
            r=s.turn('住院合规费用15000元，实际赔多少？等待期呢？')
        self.assertNotIn('3000',r.text)
        self.assertNotIn('假设',r.text)
        self.assertEqual([t['status'] for t in s.consultation['tasks']],['needs_user','needs_answer'])
        self.assertEqual(s.consultation['status'],'pending')

    def test_policy_social_identity_alias_requires_explicit_policy_user_quote(self):
        raw={'facts':[fact('has_social',True,'有医保身份投保并结算'),fact('remaining_deductible',0,'免赔已经全部抵扣')],
             'tasks':[task(1,'claim_calculation')]}
        state=validate_progress(raw,start('实际赔多少？','c1'),['有医保身份投保并结算，免赔已经全部抵扣'],{},'实际赔多少？')
        self.assertEqual(state['status'],'complete')
        self.assertIn('insured_with_social',[f['name'] for f in state['facts']])

    def test_price_dependency_omission_cannot_mark_actual_quote_complete(self):
        raw={'facts':[fact('age',35,'35岁')],'tasks':[task(1,'price')]}
        state=validate_progress(raw,start('35岁多少钱？','c1'),['35岁多少钱？'],{'age':35},'35岁多少钱？')
        self.assertEqual(state['status'],'pending')
        self.assertIn('has_social',state['tasks'][0]['missing_conditions'])

    def test_consent_with_unknown_program_quote_scope_stops_after_one_question(self):
        s,calls=self.session([],[{'relation':'本人','age':35,'has_social':True}])
        with patch('agent.small_model.classify_act',return_value='同意投保'):
            r=s.turn('我35岁有医保，发链接吧。')
        self.assertEqual(len(calls),0)
        self.assertEqual(r.text.count('首次投保必选计划个人单'),1)
        self.assertFalse(s.flags['link'])

    def test_explicitly_unavailable_quote_does_not_require_unrelated_social_identity(self):
        raw={'facts':[fact('age',72,'72岁')],
             'tasks':[{**task(1,'price'),'quoted_amount':None}]}
        state=validate_progress(raw,start('父亲72岁第一次买，多少钱？','c1'),['父亲72岁第一次买，多少钱？'],{'age':72},'多少钱？')
        self.assertEqual(state['status'],'complete')

    def test_actual_discount_keeps_family_count_pending_after_general_rule_branches(self):
        raw={'facts':[],'tasks':[task(1,'discount')]}
        state=validate_progress(raw,start('家庭买有多少优惠？','c1'),['家庭买有多少优惠？'],{},'家庭买有多少优惠？')
        self.assertEqual(state['status'],'pending')
        self.assertIn('family_size',state['tasks'][0]['missing_conditions'])

    def test_new_insured_person_cannot_inherit_previous_person_social_identity(self):
        s,calls=self.session([output('请问母亲有医保吗？',[task(1,'price','needs_user',['age','has_social'])],
                                    [fact('age',65,'母亲65岁'),fact('has_social',None,state='missing')],'clarify')],
                             [{'relation':'父母','age':65,'gender':'女'}])
        s.profile={'relation':'父母','age':60,'has_social':True,'gender':'男'}
        s.flags['consent']=True
        s.history=[('用户','父亲60岁，有职工医保')]
        s.consultation={**start('父亲年费多少？','old'),'status':'complete'}
        with patch('agent.small_model.classify_act',return_value='提问'):
            r=s.turn('那改成母亲65岁首次买必选计划个人单年缴，父亲不买。')
        self.assertNotIn('has_social',s.profile)
        self.assertFalse(s.flags['consent'])
        self.assertIn('有基本医保',r.text)
        self.assertNotEqual(s.consultation['id'],'old')

    def test_previous_person_quote_cannot_confirm_current_person_fact(self):
        raw={'facts':[fact('has_social',True,'父亲有医保'),fact('age',65,'母亲65岁')],
             'tasks':[task(1,'price')]}
        state=validate_progress(raw,start('母亲报价多少？','c1'),['父亲有医保','改成母亲65岁'],{'age':65},
                                '改成母亲65岁',condition_user_texts=['改成母亲65岁'])
        self.assertEqual(state['tasks'][0]['status'],'needs_user')
        self.assertIn('has_social',state['tasks'][0]['missing_conditions'])

    def test_allowance_and_pharmacy_calculations_do_not_use_medical_deductible(self):
        for source in ('R14:D116','R15:D118','R16:D122'):
            raw={'facts':[],'tasks':[task(1,'claim_calculation')]}
            state=validate_progress(raw,start('津贴或药省保算多少？','c1'),['津贴或药省保算多少？'],{},'',answer_source_ids=[source])
            self.assertEqual(state['status'],'complete')
            self.assertEqual(state['tasks'][0]['requires'],[])

    def test_gender_based_addon_price_does_not_ask_irrelevant_social_identity(self):
        raw={'facts':[fact('age',31,'31岁女性')],
             'tasks':[{**task(1,'price'),'question':'31岁女性重疾关爱加油包每5万价格'}]}
        state=validate_progress(raw,start('31岁女性重疾关爱加油包价格？','c1'),['31岁女性重疾关爱加油包价格？'],{'age':31},'',answer_source_ids=['rate:年缴_加油包:9'])
        self.assertEqual(state['status'],'complete')
        self.assertNotIn('has_social',state['tasks'][0]['requires'])

    def test_zero_deductible_outpatient_addon_does_not_ask_remaining_annual_deductible(self):
        raw={'facts':[fact('insured_with_social',True,'有医保身份投保')],'tasks':[task(1,'claim_calculation')]}
        state=validate_progress(raw,start('有医保身份投保，门急诊加油包赔多少？','c1'),['有医保身份投保，门急诊加油包赔多少？'],{},'',answer_source_ids=['R13:D113'])
        self.assertEqual(state['status'],'complete')

    def test_confirmed_social_settlement_does_not_require_enrollment_identity_again(self):
        raw={'facts':[fact('social_settled',True,'已医保结算'),fact('remaining_deductible',10000,'年免赔还没用')],
             'tasks':[task(1,'claim_calculation')]}
        state=validate_progress(raw,start('已医保结算，年免赔还没用，赔多少？','c1'),['已医保结算，年免赔还没用，赔多少？'],{},'',answer_source_ids=['R01:D8'])
        self.assertEqual(state['status'],'complete')

    def test_existing_purchase_question_is_not_purchase_consent(self):
        s,calls=self.session([output('指定名单以外不属于这项医院范围。',[task(1)])],[{}])
        with patch('agent.small_model.classify_act',return_value='同意投保'):
            r=s.turn('我买了重疾特需加油包，不在名单的私立医院VIP部能报吗？')
        self.assertEqual(len(calls),1)
        self.assertFalse(s.flags['consent'])
        self.assertIn('名单',r.text)

    def test_explicit_question_is_not_lost_to_backchannel_misclassification(self):
        s,calls=self.session([output('重投费率要按适用口径核实。',[task(1)])],[{}])
        with patch('agent.small_model.classify_act',return_value='附和'):
            r=s.turn('去年买了门急诊加油包，今年按指定期限重投还能按这个费率买吗？')
        self.assertEqual(len(calls),1)
        self.assertIn('费率',r.text)

    def test_independent_three_question_plan_cannot_be_reduced_to_two_tasks(self):
        previous=start('住院、靶向药、质子各怎么赔？','c1')
        add_questions(previous,['住院怎么赔','靶向药怎么赔','质子怎么赔'])
        raw={'facts':[],'tasks':[{**t,'status':'answered'} for t in previous['tasks'][:2]]}
        with self.assertRaisesRegex(ValueError,'tasks dropped'):
            validate_progress(raw,previous,[previous['initial_question']],{},previous['initial_question'])

    def test_missing_old_deductible_does_not_block_confirmed_new_prices(self):
        raw=output('占位',[task(1,'claim_calculation','needs_user'),task(2,'price',requires=['age','has_social'],amount=1330)],
                   [fact('age',40,'40岁'),fact('has_social',False,'没有社保')],action='clarify')
        raw['segments']=[{'text':'假设免赔一万，可赔2000元。','source_ids':['d'],'task_ids':['q1'],'derivation':''},
                         {'text':'年缴1330元、月缴131元。','source_ids':['d'],'task_ids':['q2'],'derivation':''}]
        s,_=self.session([raw],[{}])
        s.profile={'age':40,'has_social':False}
        s.history=[('用户','我40岁，没有社保。')]
        s.consultation={**start('旧住院例子还赔多少？','old'),
                        'tasks':[task(1,'claim_calculation','needs_user')]}
        with patch('agent.small_model.classify_act',return_value='提问'):
            r=s.turn('另外首次必选个人单年缴和月缴分别多少钱？')
        self.assertIn('1330',r.text)
        self.assertNotIn('2000',r.text)
        self.assertIn('免赔',r.text)
        self.assertEqual(s.consultation['tasks'][1]['status'],'answered')
        self.assertEqual(s.consultation['status'],'pending')
        self.assertTrue(s.flags['quoted'])
        self.assertFalse(s.flags['link'])

    def test_negative_social_settlement_is_not_treated_as_missing(self):
        raw={'facts':[fact('social_settled',False,'没按医保结算'),fact('insured_with_social',True,'有医保身份投保')],
             'tasks':[task(1,'claim_calculation',requires=['social_settled'])]}
        state=validate_progress(raw,start('有医保身份投保却没按医保结算，门急诊包赔多少？','c1'),
                                ['有医保身份投保却没按医保结算，门急诊包赔多少？'],{},'',answer_source_ids=['R13:D113'])
        self.assertEqual(state['status'],'complete')


if __name__=='__main__':unittest.main()
