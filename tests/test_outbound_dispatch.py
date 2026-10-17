import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agent.session import CallSession
from agent import cloud
from demo.outbound import select_action, script_remainder
from demo.speech import PRODUCT_INTRO


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.session=CallSession(None)
        self.session.history=[('坐席','我接着给您讲讲，普通住院和大病治疗分别怎么保障。'),
                              ('用户','住院怎么赔？')]
        self.client=SimpleNamespace(with_options=lambda **kw:self.client)

    def test_small_model_receives_confirmed_history_pending_tasks_and_interrupted_script(self):
        records=[{'script_id':'product_intro','response':PRODUCT_INTRO,'heard':'好的。','cancelled':True}]
        with patch('demo.outbound.cloud.chat',return_value=json.dumps({'tool':'play_script','script_id':'coverage_detail'})) as model:
            selected=select_action('你讲讲吧。',self.session,records,0,self.client)
        self.assertEqual(selected['script_id'],'coverage_detail')
        self.assertEqual(selected['source'],'small_model')
        self.assertEqual(model.call_args.kwargs['model'],cloud.FAST_MODEL)
        payload=model.call_args.args[0][1]['content']
        self.assertIn('完整已播对话',payload)
        self.assertIn('cancelled',payload)
        self.assertIn('completed_step',payload)

    def test_unknown_tools_scripts_or_history_indices_cannot_trigger_speech(self):
        for action in ({'tool':'invent_answer'}, {'tool':'play_script','script_id':'fake'},
                       {'tool':'recall_question','question_index':999}):
            with self.subTest(action=action),patch('demo.outbound.cloud.chat',return_value=json.dumps(action)):
                self.assertEqual(select_action('嗯',self.session,[],0,self.client)['tool'],'clarify')

    def test_pending_consultation_cannot_be_replaced_with_a_sales_script(self):
        self.session.consultation={'status':'pending','tasks':[{'question':'住院怎么赔'}]}
        with patch('demo.outbound.cloud.chat',return_value=json.dumps({'tool':'play_script','script_id':'product_intro'})):
            self.assertEqual(select_action('嗯',self.session,[],0,self.client)['tool'],'answer_question')

    def test_resume_uses_played_sentences_never_all_generated_text(self):
        records=[{'script_id':'product_intro','response':PRODUCT_INTRO,'heard':'好的。'}]
        self.assertEqual(script_remainder('product_intro',records),PRODUCT_INTRO[len('好的。'):])
        self.assertEqual(script_remainder('product_intro',records,True),PRODUCT_INTRO)

    def test_dispatch_failure_clarifies_instead_of_restarting_sales(self):
        with patch('demo.outbound.cloud.chat',side_effect=TimeoutError):
            result=select_action('你讲讲吧',self.session,[],0,self.client)
        self.assertEqual(result['tool'],'clarify')
        self.assertEqual(result['source'],'dispatcher_unavailable')


if __name__=='__main__':unittest.main()
