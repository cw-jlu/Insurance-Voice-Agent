import base64
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agent.cloud import RealtimeTTS

class AudioObserverTest(unittest.TestCase):
    def test_audio_observer_gets_original_packets_before_response_done(self):
        tts=RealtimeTTS.__new__(RealtimeTTS)
        tts._events_lock=threading.Lock();tts._audio=bytearray()
        tts._first_audio_t=None;tts._done=threading.Event()
        packets=[];tts._audio_consumer=lambda p:packets.append((p,tts._done.is_set()))
        for packet in [b'\x00\x01',b'\x02\x03']:
            tts._on_event({'type':'response.audio.delta','delta':base64.b64encode(packet).decode()})
        self.assertEqual(packets,[(b'\x00\x01',False),(b'\x02\x03',False)])
        self.assertEqual(bytes(tts._audio),b'\x00\x01\x02\x03')
        self.assertIsNotNone(tts._first_audio_t)
        tts._on_event({'type':'response.done'})
        self.assertTrue(tts._done.is_set())

if __name__=='__main__':unittest.main()
