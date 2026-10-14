import sys
import unittest
import threading
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agent.fast_lane import QUERY_ACK, HANDOFF_ACK, CachedSpeech, rule_ack
from agent.session import CallSession


class FastLaneTest(unittest.TestCase):
    def test_small_model_ack_does_not_wait_for_profile_understanding(self):
        pipe = Mock()
        pipe.post_answer_routing = True
        emitted = threading.Event()
        def understand(*args):
            if not emitted.wait(2):
                raise AssertionError('Acknowledgment waited for understanding')
            return {'intent': '保障', 'slots': {}, 'items': [], 'topics': [], 'gaps': []}
        pipe.understand.side_effect = understand
        pipe.retrieve.return_value = []
        from agent.pipeline import Answer
        pipe.answer.return_value = Answer('请问想了解哪项保障？', route='证据-clarify', outcome={'action': 'clarify'})
        spoken = []
        def emit(text):
            spoken.append(text)
            emitted.set()
        with patch('agent.small_model.classify_act', return_value='提问'):
            result = CallSession(pipe).turn('癌症能报不', on_sentence=emit)
        self.assertEqual(result.timings['fast_lane']['source'], 'small_model_act')
        self.assertEqual(spoken[0], QUERY_ACK)

    def test_ack_is_emitted_before_understanding_and_not_duplicated(self):
        pipe = Mock()
        pipe.post_answer_routing = True
        spoken = []
        def understand(*args):
            self.assertEqual(spoken, [QUERY_ACK])
            return {'intent': '保障', 'slots': {}, 'items': [], 'topics': [], 'gaps': []}
        pipe.understand.side_effect = understand
        pipe.retrieve.return_value = []
        from agent.pipeline import Answer
        pipe.answer.return_value = Answer('请问您想核对哪一项？', route='证据-clarify', outcome={'action': 'clarify'})
        with patch('agent.small_model.classify_act', return_value='提问'):
            result = CallSession(pipe).turn('保障能不能报？', on_sentence=spoken.append)
        self.assertEqual(spoken.count(QUERY_ACK), 1)
        self.assertEqual(result.text, ''.join(spoken))
        self.assertFalse(any(e.startswith('能力护栏:') for e in result.events))

    def test_explicit_transfer_skips_all_models_and_records_request(self):
        pipe = Mock()
        session = CallSession(pipe)
        result = session.turn('帮我转人工客服')
        pipe.understand.assert_not_called()
        pipe.answer.assert_not_called()
        self.assertEqual(result.text, HANDOFF_ACK)
        self.assertEqual(session.handoff_request['status'], 'requested')
        self.assertTrue(session.ended)

    def test_negated_and_hypothetical_transfer_does_not_execute(self):
        for text in ['不要转人工', '如果转人工会怎么样？', '转人工是什么意思？',
                     '你是人工客服吗？', '有没有人工客服？', '我不想转人工']:
            self.assertNotEqual(rule_ack(text)[0], 'handoff')

    def test_explicit_consultation_survives_small_model_misclassification(self):
        self.assertEqual(rule_ack('展，了解癌症相关保障。')[0], 'query')
        self.assertEqual(rule_ack('我已经了解保障了')[1], '')
        pipe = Mock()
        pipe.post_answer_routing = True
        pipe.understand.return_value = {'intent': '保障', 'slots': {}, 'items': [], 'topics': [], 'gaps': []}
        pipe.retrieve.return_value = []
        from agent.pipeline import Answer
        pipe.answer.return_value = Answer('请问想了解哪项保障？', route='证据-clarify', outcome={'action': 'clarify'})
        session = CallSession(pipe)
        with patch('agent.small_model.classify_act', return_value='附和'):
            result = session.turn('想了解癌症相关保障')
        pipe.answer.assert_called_once()
        self.assertFalse(session.flags['link'])
        self.assertEqual(result.act, '提问')

    def test_cache_delivers_original_audio_without_extra_request(self):
        tts = Mock()
        tts.synth.return_value = (0.2, 0.3, b'pcm')
        speech = CachedSpeech(tts)
        speech.warm()
        count = tts.synth.call_count
        packets = []
        speech.synth(QUERY_ACK, on_audio=packets.append)
        self.assertEqual(tts.synth.call_count, count)
        self.assertEqual(packets, [b'pcm'])


if __name__ == '__main__':
    unittest.main()
