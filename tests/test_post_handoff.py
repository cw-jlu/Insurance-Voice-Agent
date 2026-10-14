import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agent.evidence_answer import Evidence, EvidenceAnswerer
from agent.evidence_pipeline import EvidencePipeline
from agent.handoff import validate_outcome
from agent.session import CallSession, POST_HANDOFF


class PostHandoffTest(unittest.TestCase):
    def setUp(self):
        self.evidence=[Evidence('materials:manifest','本轮未提供完整药品清单附件','目录','p','h'),
            Evidence('d','每年318元','费率','p','h')]

    def output(self,action='handoff'):
        return {'segments':[{'text':'本轮没有完整药品清单，需要转人工核实。',
            'source_ids':['materials:manifest'],'derivation':''}],
            'unknowns':[], 'outcome':{'action':action,
            'reason_kind':'missing_material' if action=='handoff' else 'missing_user_condition',
            'reason':'缺完整清单' if action=='handoff' else '缺年龄',
            'pending_questions':['指定药是否在清单内'],'source_ids':['materials:manifest']}}

    def test_one_call_preserves_model_text_and_creates_traceable_request(self):
        engine=EvidenceAnswerer(client=object())
        output=self.output()
        with patch.object(engine,'call',return_value={'model':'test','output':output,'seconds':0}) as call:
            result=engine.answer('指定药能报吗',{'age':30},self.evidence,allow_handoff=True)
        self.assertEqual(call.call_count,1)
        self.assertEqual(result['text'],output['segments'][0]['text'])
        self.assertTrue(result['provenance']['valid'])
        self.assertEqual(result['handoff_request']['status'],'requested')
        self.assertEqual(result['handoff_request']['known'],{'age':30})
        self.assertEqual(result['handoff_request']['claims'][0]['citations'][0]['quote'],self.evidence[0].text)

    def test_invalid_or_unsubstantiated_action_is_rejected_without_repair(self):
        out=self.output()['outcome']
        out['source_ids']=['fake']
        with self.assertRaises(ValueError):validate_outcome(out,self.evidence)
        out=self.output()['outcome'];out['reason_kind']='source_conflict'
        with self.assertRaises(ValueError):validate_outcome(out,self.evidence)
        out=self.output()['outcome'];out['action']='answer';out['reason_kind']='none'
        with self.assertRaises(ValueError):validate_outcome(out,self.evidence)

    def test_invalid_generation_is_retained_but_not_executed_or_spoken(self):
        out=self.output();out['outcome']['source_ids']=['fake']
        s,engine=self.session(out)
        spoken=[]
        with patch.object(engine,'call',return_value={'model':'test','output':out,'seconds':0}), \
             patch('agent.small_model.classify_act',return_value='提问'):
            result=s.turn('指定药能报吗',on_sentence=spoken.append)
        self.assertFalse(s.flags['handoff'])
        self.assertFalse(s.flags['link'])
        self.assertFalse(s.ended)
        self.assertEqual(result.text,'本次回答暂时未能完成，请稍后再试。')
        self.assertNotIn(out['segments'][0]['text'],spoken)
        raw=result.timings['answer_detail']['evidence_result']
        self.assertEqual(raw['outcome']['source_ids'],['fake'])
        self.assertFalse(raw['outcome_validation']['valid'])
        self.assertIsNone(raw['handoff_request'])

    def session(self,output):
        engine=EvidenceAnswerer(client=object())
        pipe=EvidencePipeline(self.evidence,client=object(),answerer=engine,consultation_tracking=False)
        pipe.understand=lambda *a:{'intent':'保障','slots':{},'items':[],'topics':[],'gaps':[]}
        pipe.retrieve=lambda *a:[]
        s=CallSession(pipe,profile={'age':30,'has_social':True},voice=False)
        return s,engine

    def test_invalid_claim_reference_does_not_release_or_transfer(self):
        out=self.output();out['segments'][0]['source_ids']=['fake']
        s,engine=self.session(out)
        spoken=[]
        with patch.object(engine,'call',return_value={'model':'test','output':out,'seconds':0}), \
             patch('agent.small_model.classify_act',return_value='提问'):
            result=s.turn('指定药能报吗',on_sentence=spoken.append)
        self.assertFalse(s.flags['handoff'])
        self.assertFalse(s.flags['link'])
        self.assertNotIn(out['segments'][0]['text'],spoken)
        self.assertEqual(result.text,'本次回答暂时未能完成，请稍后再试。')
        raw=result.timings['answer_detail']['evidence_result']
        self.assertTrue(raw['outcome_validation']['valid'])
        self.assertFalse(raw['provenance']['valid'])

    def test_post_handoff_stops_sales_and_link_and_emits_only_once(self):
        out=self.output();s,engine=self.session(out)
        s.flags.update(quoted=True,limits=True,disclosure=True)
        spoken=[]
        with patch.object(engine,'call',return_value={'model':'test','output':out,'seconds':0}), \
             patch('agent.small_model.classify_act',return_value='提问'), \
             patch('agent.small_model.detect_gaps',return_value=[]):
            result=s.turn('指定药能不能报',on_sentence=spoken.append)
        self.assertTrue(s.flags['handoff'])
        self.assertFalse(s.flags['link'])
        self.assertTrue(s.ended)
        self.assertIn('后置转人工请求',result.events)
        self.assertEqual(spoken,[out['segments'][0]['text'],POST_HANDOFF])
        self.assertEqual(result.text,''.join(spoken))
        self.assertEqual(s.handoff_request['question'],'指定药能不能报')

    def test_clarification_does_not_transfer_or_resume_sales(self):
        out=self.output('clarify');out['segments'][0]['text']='请问被保人几岁？'
        s,engine=self.session(out)
        with patch.object(engine,'call',return_value={'model':'test','output':out,'seconds':0}), \
             patch('agent.small_model.classify_act',return_value='提问'), \
             patch('agent.small_model.detect_gaps',return_value=[]):
            result=s.turn('基础多少钱？')
        self.assertFalse(s.flags['handoff'])
        self.assertFalse(s.ended)
        self.assertEqual(result.text,'请问被保人几岁？')
        self.assertIn('追问缺少条件',result.events)

    def test_explicit_human_request_is_preemptive_without_answer_call(self):
        s,engine=self.session(self.output())
        with patch.object(engine,'call') as call, \
             patch('agent.small_model.classify_act',return_value='提问'), \
             patch('agent.small_model.detect_gaps',return_value=[]):
            result=s.turn('帮我转人工客服')
        call.assert_not_called()
        self.assertTrue(s.flags['handoff'])
        self.assertIn('转人工',result.events)

if __name__=='__main__':unittest.main()
