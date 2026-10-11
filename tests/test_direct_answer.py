import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agent.pipeline import Pipeline, understand_split, item_chunks
from agent.cloud import StreamResult

class DirectAnswerTest(unittest.TestCase):
    def test_flow_extraction_has_no_answer_planning_request(self):
        client = SimpleNamespace(with_options=lambda **kw: object())
        with patch('agent.pipeline.cloud.chat', return_value='{"intent":"保障","gaps":[]}') as chat, \
             patch('agent.small_model.extract_slots', return_value={'age': 30}):
            u = understand_split('住院和门诊怎么报', {}, client=client, answer_planning=False)
        self.assertEqual(chat.call_count, 1)
        self.assertNotIn('obligations', u)
        self.assertEqual(u['slots']['age'], 30)

    def test_direct_model_receives_original_question_without_slots(self):
        question = '我有社保，门诊加油包和基础版的门急诊有什么区别'
        u = {'intent':'保障', 'slots':{'has_social':True}, 'gaps':[], 'items':['R02','R13'],
             'topics':[], 'obligations':['虚构的医院问题'], '_voice_questions':['虚构问题'],
             '_retrieved':item_chunks('R02') + item_chunks('R13')}
        with patch('agent.pipeline.cloud.chat_stream', return_value=StreamResult('基础版有免赔额；加油包需另购。', .1, .2)) as chat, \
             patch('agent.pipeline._guard', return_value=[]):
            a = Pipeline(None, client=object(), direct_answer=True).answer(question, voice=True, understanding=u)
        messages = chat.call_args.args[0]
        self.assertIn(question, messages[1]['content'])
        self.assertNotIn('逐项回答槽位', messages[1]['content'])
        self.assertNotIn('虚构问题', messages[1]['content'])
        self.assertNotIn('response_format', chat.call_args.kwargs)
        self.assertNotIn('obligations', a.understanding)
        self.assertEqual(a.timings['trace'][0]['mode'], 'direct')
        self.assertEqual(u['obligations'], ['虚构的医院问题'])

if __name__ == '__main__':
    unittest.main()
