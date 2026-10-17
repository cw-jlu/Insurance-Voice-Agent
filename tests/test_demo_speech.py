import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from demo.speech import SpeechCache, SpeechServiceError, WebRealtimeTTS
from demo.service import DemoSession, fixed_speech
from demo.telemetry import Timeline


class DemoSpeechTests(unittest.TestCase):
    def test_native_https_total_budget_does_not_add_three_timeouts(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
        from windows_https_transport import WindowsHttpsTransport
        from types import SimpleNamespace
        import httpx
        request=httpx.Request('POST','https://example.com/',content='{}',
                             extensions={'timeout':{'connect':30,'read':30,'write':30}})
        completed=SimpleNamespace(returncode=0,stdout=b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}')
        with patch('windows_https_transport.subprocess.run',return_value=completed) as run:
            response=WindowsHttpsTransport(total_timeout=12).handle_request(request)
        args=run.call_args.args[0]
        self.assertEqual(args[args.index('--max-time')+1],'12')
        self.assertEqual(args[args.index('--connect-timeout')+1],'12')
        self.assertEqual(response.status_code,200)

    def test_tts_waits_for_both_server_acknowledgements_before_sending_text(self):
        order=[]
        events=iter([
            {'type':'session.created'}, {'type':'session.updated'},
            {'type':'response.audio.delta','delta':base64.b64encode(b'\x01\x00'*100).decode()},
            {'type':'response.done'}])
        class Socket:
            def recv(self,timeout):
                value=next(events);order.append(value['type']);return json.dumps(value)
            def send(self,raw):order.append(json.loads(raw)['type'])
            def close(self):pass
        with patch('websockets.sync.client.connect',return_value=Socket()):
            tts=WebRealtimeTTS();first,total,pcm=tts.synth('测试')
        self.assertEqual(order[:5],['session.created','session.update','session.updated',
                                    'input_text_buffer.append','input_text_buffer.commit'])
        self.assertEqual(len(pcm),200)
        self.assertLessEqual(first,total)

    def test_service_quota_error_fails_before_any_text_is_submitted(self):
        class Socket:
            closed=False
            def recv(self,timeout):return json.dumps({'type':'error','error':{'code':'insufficient_quota'}})
            def send(self,raw):raise AssertionError('Must not submit text after service error')
            def close(self):self.closed=True
        socket=Socket()
        with patch('websockets.sync.client.connect',return_value=socket):
            with self.assertRaises(SpeechServiceError) as caught:WebRealtimeTTS()
        self.assertEqual(caught.exception.code,'insufficient_quota')
        self.assertTrue(socket.closed)

    def test_prepared_opening_survives_restart_and_never_connects_to_tts(self):
        phrase=fixed_speech()[0];pcm=b'\x01\x00'*100
        with tempfile.TemporaryDirectory() as directory:
            SpeechCache(directory).put(phrase,pcm)
            session=DemoSession(speech_cache=SpeechCache(directory))
            timeline=Timeline()
            with patch('demo.service.WebRealtimeTTS',side_effect=AssertionError('Opening must be local')):
                self.assertEqual(session.synthesize(phrase,lambda:None,timeline,0),pcm)
            self.assertTrue(timeline.snapshot()[0]['cache_hit'])
            with patch('agent.cloud.TTS_VOICE','different-voice'):
                self.assertIsNone(session.speech_cache.get(phrase))

    def test_unprepared_fixed_speech_falls_back_immediately_instead_of_cloud_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            session=DemoSession(speech_cache=SpeechCache(directory))
            with patch('demo.service.WebRealtimeTTS',side_effect=AssertionError('No synchronous preparation')):
                with self.assertRaisesRegex(RuntimeError,'not prepared'):
                    session.synthesize(fixed_speech()[0],lambda:None,Timeline(),0)

    def test_prewarming_only_synthesizes_missing_phrases(self):
        synthesized=[]
        class TTS:
            def synth(self,text):synthesized.append(text);return 0,0,b'\x01\x00'
            def close(self):pass
        with tempfile.TemporaryDirectory() as directory:
            cache=SpeechCache(directory);cache.put('已有',b'\x01\x00')
            cache.warm(['已有','新句','新句'],factory=TTS)
            self.assertEqual(synthesized,['新句'])
            self.assertEqual(cache.status,'ready')


if __name__=='__main__':unittest.main()
