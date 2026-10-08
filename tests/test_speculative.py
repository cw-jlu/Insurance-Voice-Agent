"""Speculation must never leak stale speech or change live call state before confirmation."""
import sys
import threading
import unittest
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent.speculative import SpeculativeTurn, transcript_key


@dataclass
class FakeSession:
    pipe: object = None
    history: list = field(default_factory=list)

    def turn(self, text, on_sentence=None, cancel_check=None):
        if cancel_check:
            cancel_check()
        self.history.append(text)
        if on_sentence:
            on_sentence(text + "。")
        return type("Result", (), {"text": text + "。"})()


@dataclass
class SlowSession(FakeSession):
    def turn(self, text, on_sentence=None, cancel_check=None):
        if text.startswith("旧"):
            self.pipe["started"].set()
            self.pipe["release"].wait(timeout=2)
            if cancel_check:
                cancel_check()
        return super().turn(text, on_sentence, cancel_check)


class SpeculativeTest(unittest.TestCase):
    def test_asr_formatting_normalization_preserves_conditions_and_decimals(self):
        self.assertEqual(transcript_key("我三十岁有社保多少钱"), transcript_key("我30岁，有社保，多少钱？"))
        self.assertNotEqual(transcript_key("1.5万元"), transcript_key("15万元"))
        self.assertNotEqual(transcript_key("有社保"), transcript_key("没有社保"))

    def test_exact_final_reuses_without_early_state_or_speech(self):
        live = FakeSession()
        turn = SpeculativeTurn(live, min_chars=1)
        self.addCleanup(turn.close)
        turn.partial("住院怎么报？")
        turn.candidate.future.result(timeout=2)
        self.assertEqual(live.history, [])
        spoken = []
        result = turn.confirm("住院怎么报？", spoken.append)
        self.assertTrue(turn.stats["reused"])
        self.assertEqual(spoken, [result.text])
        self.assertEqual(live.history, ["住院怎么报？"])

    def test_correction_discards_old_audio_and_old_profile(self):
        live = FakeSession()
        turn = SpeculativeTurn(live, min_chars=1, min_interval_s=0)
        self.addCleanup(turn.close)
        turn.partial("有社保")
        turn.candidate.future.result(timeout=2)
        turn.partial("没有社保")
        spoken = []
        turn.confirm("没有社保", spoken.append)
        self.assertEqual(spoken, ["没有社保。"])
        self.assertEqual(live.history, ["没有社保"])
        self.assertEqual(turn.stats["invalidations"], 1)

    def test_final_mismatch_recomputes_and_work_is_bounded(self):
        turn = SpeculativeTurn(FakeSession(), min_chars=1, min_interval_s=0, max_preparations=1)
        self.addCleanup(turn.close)
        turn.partial("三十岁")
        turn.partial("六十岁")
        spoken = []
        turn.confirm("六十一岁", spoken.append)
        self.assertEqual(spoken, ["六十一岁。"])
        self.assertFalse(turn.stats["reused"])
        self.assertEqual(turn.stats["preparations"], 1)

    def test_final_turn_does_not_wait_for_cancelled_cloud_work(self):
        gates = {"started": threading.Event(), "release": threading.Event()}
        live = SlowSession(gates)
        turn = SpeculativeTurn(live, min_chars=1, min_interval_s=0)
        self.addCleanup(turn.close)
        self.addCleanup(gates["release"].set)
        turn.partial("旧条件有社保")
        self.assertTrue(gates["started"].wait(timeout=1))
        spoken = []
        turn.confirm("新条件没有社保", spoken.append)
        self.assertFalse(gates["release"].is_set())
        self.assertEqual(spoken, ["新条件没有社保。"])
        self.assertEqual(live.history, ["新条件没有社保"])


if __name__ == "__main__":
    unittest.main()
