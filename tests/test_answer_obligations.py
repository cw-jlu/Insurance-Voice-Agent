import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agent.pipeline import Pipeline, hospital_calculation, scoped_rule_reply, obligation_guard, combine_gaps
from agent.guardrail import extract_numbers
from agent.session import CallSession

class AnswerObligationsTest(unittest.TestCase):
    def test_money_shorthand_and_conditioned_calculation(self):
        for text in ('一万五元', '1万5元'):
            self.assertEqual([n.value for n in extract_numbers(text)], [15000])
        calc = hospital_calculation('住院花了一万五，自己出多少，没走医保会怎样', {}, ['R01'])
        self.assertEqual(calc['numbers'], [5000, 3000, 10000, 12000])
        self.assertIn('假设', calc['text'])
        self.assertIsNone(hospital_calculation('住院花两万元，已经用了免赔额', {}, ['R01']))
        self.assertIsNone(hospital_calculation('住院花两万元，ICU津贴怎么算', {}, ['R01', 'R15']))

    def test_mixed_unknown_does_not_swallow_known_and_extra_questions(self):
        u = {'gaps': ['疾病核保'], 'obligations': ['高血压住院报吗', '意外住院报吗']}
        reply = scoped_rule_reply('我有高血压，住院和意外住院能报吗', u)[0]
        self.assertIn('既往症', reply)
        self.assertIn('意外住院不受', reply)
        self.assertEqual(obligation_guard('高血压和意外', reply, ['R01'], u['gaps']), [])
        u['obligations'].append('明年能续保吗')
        self.assertIsNone(scoped_rule_reply('高血压、意外、续保', u))

    def test_quote_difference_is_programmed_and_traced(self):
        u = {'intent': '报价', 'slots': {'age': 30, 'has_social': True}, 'items': [], 'topics': [], 'gaps': []}
        a = Pipeline(None, client=object()).answer('一年交和每月交差多少', understanding=u)
        self.assertIn('372', a.text)
        self.assertIn('54', a.text)
        calc = next(event for event in a.timings['trace'] if event['stage'] == 'calculate')
        self.assertEqual(calc['difference'], 54)

    def test_complaint_beats_refusal_but_stop_call_beats_complaint(self):
        class Pipe:
            client = None
            def understand(self, *args):
                return {'intent': '闲聊', 'slots': {}, 'items': [], 'topics': [], 'gaps': []}
            def retrieve(self, *args):
                return []
        with patch('agent.small_model.classify_act', return_value='拒绝'), patch('agent.small_model.detect_gaps', return_value=[]):
            s = CallSession(Pipe())
            result = s.turn('你们天天打电话太烦了！')
            self.assertTrue(s.flags['handoff'])
            self.assertIn('投诉优先路由', result.events)
            s = CallSession(Pipe())
            result = s.turn('你们天天打电话太烦了，不要再打电话')
            self.assertFalse(s.flags['handoff'])
            self.assertTrue(s.ended)

    def test_false_list_gap_is_rejected(self):
        self.assertNotIn('条款清单', combine_gaps(['条款清单'], ['条款清单'], '住院和门诊怎么报'))
        self.assertNotIn('社保抵扣免赔额', combine_gaps(['社保抵扣免赔额'], ['社保抵扣免赔额'], '家里4口人，有的有社保有的没有，免赔额共享和家庭优惠都能用吗'))

if __name__ == '__main__':
    unittest.main()
