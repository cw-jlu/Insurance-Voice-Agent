import json
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agent.fast_lane import waiting_reply,answer_while_waiting
from agent.consultation import add_questions,start


class WaitingReplyTest(unittest.TestCase):
    def test_new_question_after_failed_turn_gets_a_stable_unique_task_id(self):
        state=start('原始问题','call-1')
        add_questions(state,['第一个补充问题'])
        add_questions(state,['第二个补充问题'])
        ids=[task['id'] for task in state['tasks']]
        self.assertEqual(len(ids),2)
        self.assertEqual(len(set(ids)),2)

    def test_topic_reply_keeps_multiple_questions_without_any_business_answer(self):
        with patch('agent.fast_lane.cloud.chat',return_value=json.dumps({'focuses':['price','discount'],'tone':'ack'})):
            r=waiting_reply('多少钱，有优惠吗？','用户：35岁。',{},Mock())
        self.assertEqual(r['text'],'您关心的保费和优惠，我一起核对。')
        self.assertNotRegex(r['text'],r'\d|可以投保|能赔|已转接|已发送')

    def test_arbitrary_model_text_and_unasked_topics_cannot_be_spoken(self):
        for value in [{'focuses':['hospital'],'tone':'ack','text':'保证赔1000元'},
                      {'focuses':['price','price'],'tone':'ack'},
                      {'focuses':['price'],'tone':'already_answered'}]:
            with patch('agent.fast_lane.cloud.chat',return_value=json.dumps(value)):
                r=waiting_reply('保费多少钱？','',{},Mock())
            self.assertEqual(r['text'],'')

    def test_single_reply_is_released_during_pending_answer_then_answer_returns(self):
        done=threading.Event();speech=[];reply=Mock(return_value={'text':'您关心的保费，我一起核对。'})
        def answer():
            self.assertTrue(done.wait(2));return '正式答案'
        def emit(text):speech.append(text);done.set()
        result,info=answer_while_waiting(answer,reply,emit,delay_s=0)
        self.assertEqual(result,'正式答案')
        self.assertEqual(len(speech),1)
        reply.assert_called_once()
        self.assertTrue(info['emitted'])

    def test_immediate_answer_needs_no_small_model(self):
        reply=Mock();speech=[]
        result,info=answer_while_waiting(lambda:'已就绪',reply,speech.append,delay_s=60)
        self.assertEqual(result,'已就绪');reply.assert_not_called()
        self.assertEqual(speech,[])

    def test_answer_readiness_suppresses_late_small_model_reply(self):
        done=threading.Event();main_finished=threading.Event();speech=[]
        def answer():done.wait(2);main_finished.set();return '正式答案'
        def reply():done.set();main_finished.wait(2);return {'text':'迟到承接'}
        result,info=answer_while_waiting(answer,reply,speech.append,delay_s=0)
        self.assertEqual(result,'正式答案');self.assertEqual(speech,[])

    def test_cancelled_turn_never_releases_delayed_reply_or_answer(self):
        started=threading.Event();cancelled=threading.Event();release=threading.Event();speech=[]
        def reply():started.set();release.wait(2);return {'text':'旧问题承接'}
        def answer():release.wait(2);return '旧业务答案'
        def check():
            if cancelled.is_set():raise RuntimeError('cancelled')
        with ThreadPoolExecutor(1) as pool:
            task=pool.submit(answer_while_waiting,answer,reply,speech.append,check,0)
            self.assertTrue(started.wait(2));cancelled.set()
            try:
                with self.assertRaisesRegex(RuntimeError,'cancelled'):task.result(timeout=1)
            finally:release.set()
        self.assertEqual(speech,[])


if __name__=='__main__':unittest.main()
