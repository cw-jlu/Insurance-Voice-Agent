"""Current-turn task provenance and continued consultation release contracts."""
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agent.consultation import add_questions, question_plan, reopen, start, validate_progress
from agent.pipeline import understand_split
from agent.evidence_answer import Evidence, EvidenceAnswerer
from agent.evidence_pipeline import EvidencePipeline
from agent.session import CallSession


def entry(question, quote, task_id=None):
    return {'question':question,'user_quote':quote,'task_id':task_id}


def task(identifier, question, status='answered', requires=()):
    return {'id':identifier,'question':question,'kind':'coverage','status':status,
            'requires':list(requires),'is_example':False}


class QuestionPlanningTest(unittest.TestCase):
    def test_history_only_questions_do_not_enter_current_plan(self):
        current='意外住院也要等30天吗？'
        plan=question_plan([entry('靶向药怎么赔','靶向药怎么赔'),
                            entry('意外等待期',current)],current)
        self.assertEqual(plan,[entry('意外等待期',current)])

    def test_condition_only_and_invalid_planner_outputs_create_no_tasks(self):
        for value in (None,{},'月缴多少钱',['月缴多少钱'],
                      [entry('月缴多少钱','')], [entry('月缴多少钱',25)],
                      [entry(None,'有医保')], [entry('月缴多少钱','有医保',[])],
                      [entry('月缴多少钱','有医保','invented')]):
            with self.subTest(value=value):
                self.assertEqual(question_plan(value,'她有医保'),[])

    def test_compound_current_questions_are_kept_even_with_same_quote(self):
        current='住院和靶向药各怎么赔？'
        plan=question_plan([entry('住院怎么赔',current),entry('靶向药怎么赔',current)],current)
        self.assertEqual(len(plan),2)

    def test_explicit_repeat_reuses_id_and_requires_a_new_spoken_answer(self):
        state=start('月缴多少钱','c1')
        state['tasks']=[task('old','月缴保费')]
        current='刚才月缴那个再说一次。'
        add_questions(state,[entry('每个月多少钱',current,'old')],current)
        self.assertEqual(len(state['tasks']),1)
        self.assertEqual(state['tasks'][0]['id'],'old')
        self.assertEqual(state['tasks'][0]['status'],'needs_answer')

    def test_same_question_without_id_does_not_duplicate_existing_task(self):
        state=start('月缴多少钱','c1')
        state['tasks']=[task('old','月缴多少钱？')]
        add_questions(state,[entry('月缴多少钱','月缴多少钱')],'月缴多少钱')
        self.assertEqual([t['id'] for t in state['tasks']],['old'])
        self.assertEqual(state['tasks'][0]['status'],'needs_answer')

    def test_failed_revision_still_allocates_unique_ids(self):
        state=start('保障问题','c1')
        add_questions(state,[entry('等待期','等待期')],'等待期')
        add_questions(state,[entry('免赔额','免赔额')],'免赔额')
        self.assertEqual(len({t['id'] for t in state['tasks']}),2)

    def test_age_correction_reopens_dependent_prices_but_keeps_coverage_answered(self):
        state=start('价格和医院范围','c1')
        state['tasks']=[task('price','年费',requires=['age','has_social']),
                        task('hospital','医院范围'),task('cancelled','不再问的价格','cancelled',['age'])]
        state['facts']=[{'name':'age','value':30},{'name':'has_social','value':True}]
        reopen(state,['age'])
        self.assertEqual([t['status'] for t in state['tasks']],['pending','answered','cancelled'])
        self.assertEqual([f['name'] for f in state['facts']],['has_social'])

    def test_interruption_without_specific_condition_reopens_every_answered_task(self):
        state=start('原问题','c1')
        state['tasks']=[task('price','年费',requires=['age']),task('hospital','医院范围')]
        reopen(state)
        self.assertEqual([t['status'] for t in state['tasks']],['pending','pending'])

    def test_seed_provenance_survives_model_omission_across_rounds(self):
        state=start('等待期？','c1')
        add_questions(state,[entry('等待期','等待期')],'等待期')
        returned={**state['tasks'][0],'status':'answered','kind':'coverage'}
        returned.pop('seeded');returned.pop('user_quote')
        result=validate_progress({'tasks':[returned],'facts':[]},state,['等待期'],{},'等待期')
        self.assertTrue(result['tasks'][0]['seeded'])
        self.assertEqual(result['tasks'][0]['user_quote'],'等待期')

    def test_split_planner_filters_history_quotes_and_receives_existing_ids(self):
        current='意外住院也要等30天吗？'
        state=start('历史问题','c1');state['tasks']=[task('old','意外等待期')]
        prompts=[]
        def chat(messages,**kwargs):
            if 'obligations' in messages[0]['content']:
                prompts.append(messages[-1]['content'])
                return json.dumps({'items':['R01'],'topics':['等待期'],'obligations':[
                    entry('靶向药怎么赔','靶向药怎么赔'),entry('意外等待期',current,'old')]})
            return json.dumps({'intent':'保障','gaps':[]})
        with patch('agent.pipeline.cloud.chat',side_effect=chat), \
                patch('agent.small_model.extract_slots',return_value={}):
            u=understand_split(current,{'_history':'用户：靶向药怎么赔？','_consultation':state},Mock())
        self.assertEqual(u['obligations'],['意外等待期'])
        self.assertEqual(u['obligation_plan'][0]['task_id'],'old')
        self.assertIn('old',prompts[0])
        self.assertIn('【本轮用户原话】'+current,prompts[0])

    def session(self,spoken=True):
        engine=EvidenceAnswerer(client=object())
        pipe=EvidencePipeline([Evidence('d','测试免责依据','位置','path','hash')],client=object(),answerer=engine)
        current='高血压无关的意外住院一定不赔吗？'
        plan=[entry('无关意外是否赔',current)]
        pipe.understand=lambda *_:{'intent':'保障','slots':{},'gaps':[],'items':[],
                                   'topics':[],'obligations':['无关意外是否赔'],'obligation_plan':plan}
        s=CallSession(pipe,voice=True,fast_lane=False,playback_tracking=True)
        s.consultation=start('旧住院金额问题','c1')
        s.consultation['tasks']=[task('old-drug','靶向药怎么赔'),
                                task('old-claim','旧住院金额','needs_user',['remaining_deductible'])]
        def call(system,user,model):
            seeded=deepcopy(s.consultation['tasks'])
            for t in seeded:
                if t['id'].startswith('c1-u'):
                    t['status']='answered'
            segments=[{'text':'无关意外仍需按责任约定确认。','source_ids':['d'],
                       'task_ids':[t['id'] for t in seeded if t['id'].startswith('c1-u')] if spoken else []}]
            return {'model':'test','seconds':0,'output':{'segments':segments,'unknowns':[],
                'consultation':{'tasks':seeded,'facts':[]},
                'outcome':{'action':'clarify','reason_kind':'missing_user_condition','reason':'旧金额缺条件',
                           'pending_questions':['剩余免赔额'],'source_ids':[]}}}
        engine.call=call
        return s,current

    def test_new_spoken_answer_survives_pending_old_calculation_and_playback(self):
        s,current=self.session()
        with patch('agent.small_model.classify_act',return_value='提问'):
            r=s.turn(current)
        self.assertEqual(r.timings['answer_validation']['route'],'证据-clarify')
        self.assertIn('无关意外',r.text)
        self.assertEqual(len(s.consultation['tasks']),3)
        self.assertEqual(s.consultation['status'],'awaiting_playback')
        self.assertTrue(s.confirm_playback(r.playback_id,r.text,True))
        self.assertEqual(s.consultation['status'],'pending')
        self.assertEqual(s.consultation['tasks'][0]['status'],'answered')
        self.assertFalse(s.flags['link'])

    def test_genuine_new_task_without_spoken_segment_is_still_blocked(self):
        s,current=self.session(spoken=False)
        with patch('agent.small_model.classify_act',return_value='提问'):
            r=s.turn(current)
        self.assertEqual(r.timings['answer_validation']['route'],'证据-待办异常')
        self.assertNotIn('无关意外仍需',r.text)
        self.assertIn('Seeded answered tasks',r.timings['answer_detail']['consultation_fault'])


if __name__=='__main__':
    unittest.main()
