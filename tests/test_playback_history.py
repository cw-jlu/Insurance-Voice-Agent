import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agent.dialogue import PlaybackLedger, SentState
from agent.pipeline import Answer
from agent.playback import PlaybackReceipts
from agent.session import CallSession
from agent.speculative import SpeculativeTurn


class PlaybackHistoryTest(unittest.TestCase):
    def test_confirmed_speculative_state_does_not_overwrite_playback_receipts(self):
        session = self.session()
        manager = SpeculativeTurn(session, min_chars=1)
        try:
            with patch('agent.small_model.classify_act', return_value='提问'):
                manager.partial('多少钱？')
                manager.candidate.future.result(timeout=5)
                def emit(text):
                    if text == '年缴318元。':
                        session.confirm_playback(session._playback_active, text)
                manager.confirm('多少钱？', emit)
            self.assertEqual(session.history[-1], ('坐席', '年缴318元。'))
            self.assertNotIn('月缴31元', session._history_text())
        finally:
            manager.close()

    def session(self):
        pipe = Mock()
        pipe.post_answer_routing = True
        pipe.understand.return_value = {'intent': '保障', 'slots': {}, 'items': [], 'topics': [], 'gaps': []}
        pipe.retrieve.return_value = []
        pipe.answer.return_value = Answer('年缴318元。月缴31元。', route='证据-answer', outcome={'action': 'clarify'})
        return CallSession(pipe, playback_tracking=True, fast_lane=False)

    def test_next_model_sees_only_completed_speech_and_late_receipts_do_not_restore_it(self):
        session = self.session()
        with patch('agent.small_model.classify_act', return_value='提问'):
            old = session.turn('多少钱？')
            self.assertNotIn('月缴31元', session._history_text())
            receipts = PlaybackReceipts(session, old.playback_id)
            a = receipts.enqueue('年缴318元。')
            b = receipts.enqueue('月缴31元。')
            receipts.scheduled_end(a, 1.0)
            receipts.scheduled_end(b, 4.0)
            report = receipts.interrupt(now=2.0)
            self.assertEqual(report, {'heard': '年缴318元。', 'dropped': '月缴31元。'})
            session.turn('我没有社保，重新报价？')
            known = session.pipe.answer.call_args.args[1]
            self.assertIn('年缴318元。', known['_history'])
            self.assertNotIn('月缴31元', known['_history'])
            self.assertFalse(session.confirm_playback(old.playback_id, old.text, complete=True))
            self.assertEqual(session.history[1][1], '年缴318元。')

    def test_complete_speech_is_committed_and_future_dac_buffer_is_not_yet_heard(self):
        session = self.session()
        with patch('agent.small_model.classify_act', return_value='提问'):
            result = session.turn('多少钱？')
        receipts = PlaybackReceipts(session, result.playback_id)
        token = receipts.enqueue(result.text)
        receipts.scheduled_end(token, 4.0)
        self.assertEqual(receipts.flush(now=3.9), '')
        self.assertTrue(receipts.finish(now=4.0))
        self.assertEqual(session.history[-1], ('坐席', result.text))

    def test_interrupted_unaligned_sentence_is_not_marked_fully_heard(self):
        ledger = PlaybackLedger()
        sentence = ledger.add('月缴31元，还有其他条件。')
        sentence.state = SentState.PLAYING
        heard, dropped = ledger.interrupt()
        self.assertEqual(heard, [])
        self.assertEqual(dropped, [sentence.text])

    def test_confirmed_aligned_partial_prefix_is_kept(self):
        ledger = PlaybackLedger()
        sentence = ledger.add('年缴318元，月缴31元。')
        sentence.state = SentState.PLAYING
        sentence.heard_prefix = '年缴318元，'
        self.assertEqual(ledger.interrupt(), (['年缴318元，'], ['月缴31元。']))

    def test_interruption_rolls_back_unheard_disclosures(self):
        session = self.session()
        with patch('agent.small_model.classify_act', return_value='提问'):
            result = session.turn('多少钱？')
        session.flags.update(quoted=True, limits=True, disclosure=True)
        session.interrupt_playback(result.playback_id)
        self.assertFalse(session.flags['quoted'])
        self.assertFalse(session.flags['limits'])
        self.assertFalse(session.flags['disclosure'])


if __name__ == '__main__':
    unittest.main()
