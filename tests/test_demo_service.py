import asyncio
import json
import sys
import threading
import hashlib
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fastapi.testclient import TestClient
from demo.app import create_app
from demo.service import DemoSession, DemoStore, demo_text, OUTBOUND_OPENING, PRODUCT_INTRO, PRODUCT_DETAIL
from demo.source_compat import source_digest
from agent.pipeline import Answer
from agent.session import HANDOFF, LINK_SENT


def receive_until(ws,kind):
    for _ in range(100):
        value=ws.receive_json()
        if value['type']==kind:return value
    raise AssertionError('Missing event: '+kind)


class FakePipe:
    post_answer_routing=True
    consultation_tracking=True
    client=None
    def __init__(self,blocked=False):
        self.started=threading.Event();self.release=threading.Event()
        if not blocked:self.release.set()
    def retrieve(self,*args):return []
    def understand(self,*args):return {'intent':'保障','slots':{'age':35},'gaps':[],
                                      'items':[],'topics':[],'obligations':[],'obligation_plan':[]}
    def answer(self,question,known,**kwargs):
        self.started.set()
        if not self.release.wait(3):raise TimeoutError('Test worker not released')
        state=deepcopy(known['_consultation'])
        state.update(status='complete',revision=1,tasks=[{'id':'q1','question':question,
                     'kind':'coverage','status':'answered','requires':[]}])
        return Answer('第一句。第二句。',route='证据-answer',outcome={'action':'answer'},
                      timings={'total':0,'consultation':state})


class FakeStore(DemoStore):
    def __init__(self,pipe):super().__init__();self.pipe=pipe
    def create(self,mode,provider,scenario='consultation'):
        session=DemoSession(mode,provider,self.pipe,scenario)
        self.sessions[session.id]=session
        return session


class DemoServiceTest(unittest.TestCase):
    def test_outbound_opening_has_no_fabricated_user_and_requires_playback_receipts(self):
        store=FakeStore(FakePipe())
        with TestClient(create_app(store)) as client:
            state=client.post('/api/sessions',json={'mode':'live','scenario':'outbound'}).json()
            self.assertEqual(state['scenario'],'outbound')
            with client.websocket_connect('/api/sessions/'+state['id']+'/ws') as ws:
                ws.receive_json()
                ws.send_json({'type':'turn','text':'你好'})
                self.assertIn('先点击',receive_until(ws,'error')['message'])
                ws.send_json({'type':'start'})
                started=receive_until(ws,'turn_start')
                self.assertEqual(started['kind'],'opening')
                self.assertEqual(started['question'],'')
                done=receive_until(ws,'turn_done')
                self.assertEqual(done['text'],OUTBOUND_OPENING)
                self.assertIn('方便',done['text'])
                self.assertNotIn('录音',done['text'])
                self.assertNotIn('之前了解过',done['text'])
                self.assertFalse(store.sessions[state['id']].session.flags['recording'])
                self.assertTrue(done['state']['opening_started'])
                self.assertEqual(done['state']['history'],[['坐席','']])
                self.assertEqual(done['state']['turns'][0]['kind'],'opening')
                for index in range(len(store.sessions[state['id']].speech)):
                    ws.send_json({'type':'receipt','turn':started['turn'],'index':index})
                    receipt=receive_until(ws,'receipt')
                    self.assertTrue(receipt['accepted'])
                self.assertEqual(receipt['state']['history'],[['坐席',OUTBOUND_OPENING]])
                ws.send_json({'type':'start'})
                self.assertIn('已开始',receive_until(ws,'error')['message'])

    def test_outbound_stop_keeps_only_heard_opening_then_accepts_customer_reply(self):
        store=FakeStore(FakePipe())
        with patch('demo.service.select_action',return_value={'tool':'answer_question'}),patch('agent.small_model.classify_act',return_value='提供信息'),TestClient(create_app(store)) as client:
            state=client.post('/api/sessions',json={'mode':'live','scenario':'outbound'}).json()
            with client.websocket_connect('/api/sessions/'+state['id']+'/ws') as ws:
                ws.receive_json();ws.send_json({'type':'start'})
                done=receive_until(ws,'turn_done')
                session=store.sessions[state['id']]
                first=session.speech[0].text
                ws.send_json({'type':'receipt','turn':done['turn'],'index':0})
                receive_until(ws,'receipt')
                ws.send_json({'type':'stop'})
                stopped=receive_until(ws,'stopped')
                self.assertEqual(stopped['state']['history'],[['坐席',first]])
                ws.send_json({'type':'turn','text':'给自己了解'})
                reply=receive_until(ws,'turn_done')
                self.assertIn(['用户','给自己了解'],reply['state']['history'])
                self.assertEqual(reply['state']['stage'],'需求')
                self.assertIn('为自己了解',reply['text'])

    def test_willing_listener_gets_product_introduction_before_sales_questions(self):
        session=DemoSession('live',pipe=FakePipe(),scenario='outbound')
        async def send(event):pass
        def confirm():
            for index in range(len(session.speech)):self.assertTrue(session.receipt(session.turn_id,index))
        choices=[{'tool':'play_script','script_id':'product_intro'},
                 {'tool':'play_script','script_id':'coverage_detail'}]
        with patch('demo.service.select_action',side_effect=choices),patch('agent.small_model.classify_act',side_effect=AssertionError('Script dispatch must not enter business classification')):
            asyncio.run(session.run('',send,opening=True));confirm()
            asyncio.run(session.run('嗯，可以可以。',send))
            self.assertEqual(session.result.text,PRODUCT_INTRO)
            self.assertNotIn('给您自己买',session.result.text)
            self.assertEqual(session.outbound_step,0)
            confirm();self.assertEqual(session.outbound_step,1)
            asyncio.run(session.run('你继续介绍。',send))
            self.assertEqual(session.result.text,PRODUCT_DETAIL)
            confirm();self.assertEqual(session.outbound_step,2)
            self.assertFalse(session.session.flags['consent'])

    def test_interrupted_introduction_is_not_marked_delivered_and_refusal_closes(self):
        session=DemoSession('live',pipe=FakePipe(),scenario='outbound')
        async def send(event):pass
        asyncio.run(session.run('',send,opening=True))
        choices=[{'tool':'play_script','script_id':'product_intro'},
                 {'tool':'play_script','script_id':'product_intro'}, {'tool':'end_call'}]
        with patch('demo.service.select_action',side_effect=choices):
            asyncio.run(session.run('可以',send));session.receipt(session.turn_id,0)
            first=session.speech[0].text;session.stop()
            self.assertEqual(session.outbound_step,0)
            asyncio.run(session.run('继续',send))
            self.assertEqual(session.result.text,PRODUCT_INTRO[len(first):])
            session.stop()
            asyncio.run(session.run('现在不方便，不想听了。',send))
        self.assertTrue(session.session.ended)
        self.assertIn('不打扰',session.result.text)

    def test_model_can_continue_to_details_after_interrupted_intro_without_replaying_it(self):
        session=DemoSession('live',pipe=FakePipe(),scenario='outbound')
        async def send(event):pass
        choices=[{'tool':'play_script','script_id':'product_intro'},
                 {'tool':'play_script','script_id':'coverage_detail'}, {'tool':'acknowledge'}]
        asyncio.run(session.run('',send,opening=True))
        with patch('demo.service.select_action',side_effect=choices):
            asyncio.run(session.run('嗯，可以可以。',send));session.receipt(session.turn_id,0);session.stop()
            asyncio.run(session.run('你讲讲吧。',send))
            self.assertEqual(session.result.text,PRODUCT_DETAIL)
            session.stop()
            asyncio.run(session.run('嗯。',send))
            self.assertNotIn('平时小病',session.result.text)
            self.assertNotIn('普通住院方面',session.result.text)

    def test_recalling_a_question_uses_original_user_words_without_business_answering(self):
        session=DemoSession('live',pipe=FakePipe(),scenario='outbound')
        session.opening_started=True
        session.session.history=[('用户','普通住院怎么赔？'),('用户','你讲讲吧。')]
        async def send(event):pass
        with patch('demo.service.select_action',return_value={'tool':'recall_question','question_index':0}):
            asyncio.run(session.run('我刚刚的问题是什么？',send))
        self.assertEqual(session.result.text,'您刚才说的是：“普通住院怎么赔？”')

    def test_quote_intake_asks_conditions_without_main_answer_model(self):
        session=DemoSession('live',pipe=FakePipe(),scenario='outbound');session.opening_started=True
        async def send(event):pass
        with patch('demo.service.select_action',return_value={'tool':'collect_quote_conditions'}),patch('agent.small_model.classify_act',side_effect=AssertionError('Must not enter slow chain')):
            asyncio.run(session.run('嗯，具体了解一下保费。',send))
            self.assertIn('多少周岁',session.result.text)
            self.assertEqual(session.session.stage,'报价条件')
            session.stop()
            asyncio.run(session.run('女。',send))
            self.assertIn('多少周岁',session.result.text)
            self.assertNotIn('age',session.session.profile)
            session.stop()
            asyncio.run(session.run('35岁。',send))
            self.assertEqual(session.session.profile['age'],35)
            self.assertIn('基本医保',session.result.text)
            session.stop()
            asyncio.run(session.run('有医保。',send))
            self.assertTrue(session.session.profile['has_social'])
            self.assertIn('首次投保',session.result.text)

    def test_slow_answer_has_deadline_and_cannot_commit_late_state(self):
        pipe=FakePipe(blocked=True)
        session=DemoSession('live',pipe=pipe,scenario='outbound');session.opening_started=True;session.turn_timeout_s=.1
        events=[]
        async def send(event):events.append(event)
        async def scenario():
            await session.run('普通住院怎么赔？',send)
            self.assertTrue(any(e['type']=='error' and '超时' in e['message'] for e in events))
            self.assertEqual(session.session.consultation['initial_question'],'普通住院怎么赔？')
            self.assertNotIn('age',session.session.profile)
            pipe.release.set()
        with patch('demo.service.select_action',return_value={'tool':'answer_question'}),patch('agent.small_model.classify_act',return_value='提问'):
            asyncio.run(scenario())
        self.assertNotIn('age',session.session.profile)

    def test_consultation_does_not_accept_sales_opening_and_replay_rejects_outbound(self):
        store=FakeStore(FakePipe())
        with TestClient(create_app(store)) as client:
            state=client.post('/api/sessions',json={'mode':'live'}).json()
            with client.websocket_connect('/api/sessions/'+state['id']+'/ws') as ws:
                ws.receive_json();ws.send_json({'type':'start'})
                self.assertIn('主动推销场景',receive_until(ws,'error')['message'])
                self.assertFalse(store.sessions[state['id']].opening_started)
        with TestClient(create_app()) as client:
            self.assertEqual(client.post('/api/sessions',json={'mode':'replay','scenario':'outbound'}).status_code,400)
            self.assertEqual(client.post('/api/sessions',json={'scenario':'invalid'}).status_code,422)

    def test_shell_and_static_assets_do_not_expose_configuration_files(self):
        with TestClient(create_app()) as client:
            for path in ('/','/assets/app.js','/assets/style.css','/assets/microphone.js'):
                r=client.get(path);self.assertEqual(r.status_code,200)
                self.assertIn('Content-Security-Policy',r.headers)
            self.assertEqual(client.get('/assets/../.env').status_code,404)
            self.assertEqual(client.get('/.env').status_code,404)
            self.assertEqual(client.get('/api/health').json()['core_tag'],'v0.1.0-core-20261014')

    def test_sessions_are_isolated_and_invalid_modes_rejected(self):
        with TestClient(create_app()) as client:
            a=client.post('/api/sessions',json={}).json()
            b=client.post('/api/sessions',json={}).json()
            self.assertNotEqual(a['id'],b['id'])
            self.assertEqual(client.post('/api/sessions',json={'mode':'fake'}).status_code,422)
            self.assertEqual(client.delete('/api/sessions/'+a['id']).status_code,200)
            self.assertEqual(client.get('/api/sessions/'+a['id']).status_code,404)
            self.assertEqual(client.get('/api/sessions/'+b['id']).status_code,200)

    def test_cross_origin_mutations_are_rejected(self):
        with TestClient(create_app()) as client:
            self.assertEqual(client.post('/api/sessions',json={},headers={'origin':'https://unrelated.example'}).status_code,403)

    def test_replay_uses_saved_record_and_never_calls_cloud_models(self):
        with patch('agent.cloud.chat',side_effect=AssertionError('Replay called model')),TestClient(create_app()) as client:
            state=client.post('/api/sessions',json={}).json()
            with client.websocket_connect('/api/sessions/'+state['id']+'/ws') as ws:
                ws.receive_json();ws.send_json({'type':'turn','text':state['next_question']})
                done=receive_until(ws,'turn_done')
                self.assertEqual(done['trace']['recorded_case'],'T01')
                self.assertIn('318',done['text'])
                self.assertEqual(done['state']['replay_step'],1)
                export=client.get('/api/sessions/'+state['id']+'/export').json()
                self.assertNotIn('api_calls',json.dumps(export))

    def test_replay_rejects_freeform_input_without_advancing(self):
        with TestClient(create_app()) as client:
            state=client.post('/api/sessions',json={}).json()
            with client.websocket_connect('/api/sessions/'+state['id']+'/ws') as ws:
                ws.receive_json();ws.send_json({'type':'turn','text':'自由提问'})
                error=receive_until(ws,'error')
                self.assertIn('回放只能',error['message'])
                self.assertEqual(client.get('/api/sessions/'+state['id']).json()['replay_step'],0)

    def test_completed_sentence_receipts_commit_only_heard_history(self):
        store=FakeStore(FakePipe())
        with patch('agent.small_model.classify_act',return_value='提问'),TestClient(create_app(store)) as client:
            state=client.post('/api/sessions',json={'mode':'live'}).json()
            with client.websocket_connect('/api/sessions/'+state['id']+'/ws') as ws:
                ws.receive_json();ws.send_json({'type':'turn','text':'保障问题'})
                done=receive_until(ws,'turn_done');turn=done['turn']
                self.assertEqual(done['state']['consultation']['status'],'awaiting_playback')
                ws.send_json({'type':'receipt','turn':turn,'index':1})
                self.assertFalse(receive_until(ws,'receipt')['accepted'])
                ws.send_json({'type':'receipt','turn':turn,'index':0})
                first=receive_until(ws,'receipt');self.assertTrue(first['accepted'])
                self.assertEqual(first['state']['history'][-1],['坐席','我帮您查一下。'])
                ws.send_json({'type':'receipt','turn':turn,'index':1})
                first=receive_until(ws,'receipt');self.assertTrue(first['accepted'])
                self.assertEqual(first['state']['history'][-1],['坐席','我帮您查一下。第一句。'])
                ws.send_json({'type':'stop'});stopped=receive_until(ws,'stopped')
                self.assertEqual(stopped['state']['history'][-1],['坐席','我帮您查一下。第一句。'])
                self.assertEqual(stopped['state']['consultation']['status'],'pending')
                ws.send_json({'type':'receipt','turn':turn,'index':2})
                self.assertFalse(receive_until(ws,'receipt')['accepted'])

    def test_all_sentence_receipts_finish_consultation(self):
        with patch('agent.small_model.classify_act',return_value='提问'),TestClient(create_app(FakeStore(FakePipe()))) as client:
            state=client.post('/api/sessions',json={'mode':'live'}).json()
            with client.websocket_connect('/api/sessions/'+state['id']+'/ws') as ws:
                ws.receive_json();ws.send_json({'type':'turn','text':'保障问题'})
                done=receive_until(ws,'turn_done')
                for index in range(3):
                    ws.send_json({'type':'receipt','turn':done['turn'],'index':index})
                    receipt=receive_until(ws,'receipt')
                self.assertEqual(receipt['state']['consultation']['status'],'complete')
                self.assertEqual(receipt['state']['history'][-1],['坐席','我帮您查一下。第一句。第二句。'])

    def test_cancelling_inflight_answer_discards_late_profile_and_text(self):
        pipe=FakePipe(blocked=True);store=FakeStore(pipe)
        with patch('agent.small_model.classify_act',return_value='提问'),TestClient(create_app(store)) as client:
            state=client.post('/api/sessions',json={'mode':'live'}).json()
            with client.websocket_connect('/api/sessions/'+state['id']+'/ws') as ws:
                ws.receive_json();ws.send_json({'type':'turn','text':'更正前的问题'})
                receive_until(ws,'turn_start');self.assertTrue(pipe.started.wait(2))
                ws.send_json({'type':'stop'});stopped=receive_until(ws,'stopped')
                self.assertNotIn('age',stopped['state']['profile'])
                self.assertNotIn('第一句',str(stopped['state']['history']))
                self.assertEqual(stopped['state']['history'][-1],['用户','更正前的问题'])
                pipe.release.set()
                self.assertNotIn('age',client.get('/api/sessions/'+state['id']).json()['profile'])

    def test_unsupported_actions_are_requests_not_false_success(self):
        self.assertNotIn('已经发到',demo_text(LINK_SENT))
        self.assertIn('尚未接入',demo_text(LINK_SENT))
        self.assertIn('尚未实际接通',demo_text(HANDOFF))

    def test_source_text_allows_only_expected_line_ending_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.md';path.write_bytes(b'clause\ncondition\n')
            expected=hashlib.sha256(b'clause\r\ncondition\r\n').hexdigest()
            mapping={path.resolve().as_posix():expected}
            self.assertEqual(source_digest(path,mapping),expected)
            path.write_bytes(b'changed clause\ncondition\n')
            self.assertNotEqual(source_digest(path,mapping),expected)

    def test_binary_source_hash_is_never_line_ending_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.pdf';path.write_bytes(b'pdf\nbytes')
            expected=hashlib.sha256(b'pdf\r\nbytes').hexdigest()
            self.assertNotEqual(source_digest(path,{path.resolve().as_posix():expected}),expected)

    def test_windows_source_path_is_resolved_without_changing_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.xlsx';path.write_bytes(b'workbook')
            expected=hashlib.sha256(b'workbook').hexdigest()
            self.assertEqual(source_digest(str(path).replace('/','\\'),{}),expected)


if __name__=='__main__':unittest.main()
