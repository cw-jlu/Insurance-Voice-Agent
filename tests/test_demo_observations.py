import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fastapi.testclient import TestClient
from demo.app import create_app
from demo.service import DemoSession
from demo.telemetry import Timeline, ObservedClient


class ObservationsTest(unittest.TestCase):
    def test_voice_observations_are_bounded_and_do_not_mutate_business_history(self):
        session=DemoSession()
        self.assertFalse(session.voice_event(0,{'type':'invalid'}))
        self.assertFalse(session.voice_event(9,'microphone_on'))
        self.assertTrue(session.voice_event(0,'microphone_on'))
        session.cancelled.set()
        self.assertTrue(session.voice_event(0,'confirmed_interrupt','我没有医保',{'barge_confirm_s':.24,'untrusted':'x'}))
        self.assertEqual(session.snapshot()['history'],[])
        self.assertEqual(session.export()['voice_events'][-1]['metrics'],{'barge_confirm_s':.24})
        for _ in range(600):session.voice_event(0,'asr_renew')
        self.assertEqual(len(session.export()['voice_events']),512)

    def test_cancelled_synthesis_cannot_overlap_next_question_on_same_connection(self):
        session=DemoSession();entered=threading.Event();release=threading.Event();cancelled=threading.Event()
        order=[]
        def synth(text):
            order.append(text)
            if text=='old':entered.set();release.wait(2)
            return 0,0,text.encode()
        session.tts=SimpleNamespace(synth=synth)
        def check():
            if cancelled.is_set():raise RuntimeError('cancelled')
        with ThreadPoolExecutor(max_workers=2) as pool:
            old=pool.submit(session.synthesize,'old',check,Timeline(),0)
            self.assertTrue(entered.wait(1));cancelled.set()
            new=pool.submit(session.synthesize,'new',lambda:None,Timeline(),0)
            self.assertEqual(order,['old'])
            release.set()
            with self.assertRaises(RuntimeError):old.result(2)
            self.assertEqual(new.result(2),b'new')
            self.assertEqual(order,['old','new'])

    def test_failed_call_has_finished_stage_without_sensitive_arguments(self):
        def fail(**kw):raise RuntimeError('secret API failure')
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
        events=[];timeline=Timeline(events.append)
        with self.assertRaises(RuntimeError):
            ObservedClient(client,timeline).create(model='example-model',messages=[{'content':'private user'}])
        record=timeline.snapshot()[0]
        self.assertEqual(record['status'],'error')
        self.assertGreaterEqual(record['duration_s'],0)
        self.assertEqual(len(events),2)
        self.assertNotIn('private',json.dumps(events))
        self.assertNotIn('secret',json.dumps(events))

    def test_client_options_keep_collector_and_record_token_usage(self):
        result=SimpleNamespace(usage=SimpleNamespace(prompt_tokens=2,completion_tokens=3,total_tokens=5))
        base=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:result)))
        base.with_options=lambda **kw:base
        timeline=Timeline();client=ObservedClient(base,timeline).with_options(timeout=3)
        self.assertIs(client.create(model='fast'),result)
        self.assertEqual(timeline.snapshot()[0]['tokens']['total_tokens'],5)

    def test_browser_observations_cannot_change_history_or_accept_late_turns(self):
        session=DemoSession();session.turn_id=1;session.current_record={'browser_metrics':{}}
        self.assertFalse(session.browser_metric(0,{'first_sound_s':1}))
        self.assertTrue(session.browser_metric(1,{'first_sound_s':1.2,'history':'forged','rate':float('nan'),'playback_end_s':-1}))
        self.assertEqual(session.current_record['browser_metrics'],{'first_sound_s':1.2})
        self.assertEqual(session.snapshot()['history'],[])
        session.cancelled.set();self.assertFalse(session.browser_metric(1,{'first_sound_s':2}))

    def test_cancelled_turn_keeps_generated_and_heard_text_in_one_record(self):
        session=DemoSession()
        async def send(event):pass
        asyncio.run(session.run(session.replay_row()['question'],send))
        session.receipt(session.turn_id,0);session.stop();session.stop()
        self.assertEqual(len(session.export()['turns']),1)
        record=session.export()['turns'][0]
        self.assertTrue(record['cancelled'])
        self.assertIn(record['heard'],record['response'])
        self.assertLess(len(record['heard']),len(record['response']))

    def test_source_endpoint_resolves_allowlisted_full_context(self):
        with TestClient(create_app()) as client:
            state=client.post('/api/sessions',json={}).json()
            source=client.get('/api/sessions/'+state['id']+'/sources/R01:bundle')
            self.assertEqual(source.status_code,200)
            self.assertTrue(source.json()['location'])
            self.assertEqual(len(source.json()['sha256']),64)
            self.assertEqual(client.get('/api/sessions/'+state['id']+'/sources/../../.env').status_code,404)
            self.assertEqual(client.get('/api/sessions/missing/sources/R01:bundle').status_code,404)


if __name__=='__main__':unittest.main()
