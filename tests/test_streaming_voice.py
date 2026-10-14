"""Safety and timing contracts for sentence release during LLM generation."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import cloud
from agent.pipeline import Pipeline


class StreamingVoiceTests(unittest.TestCase):
    def answer_with_chunks(self, chunks):
        released = []
        observed_midstream = []

        def fake_chat_stream(messages, model, client=None, on_delta=None, **kwargs):
            text = ""
            for i, chunk in enumerate(chunks):
                text += chunk
                if on_delta is not None:
                    on_delta(chunk, i * 0.1)
                observed_midstream.append(list(released))
            return cloud.StreamResult(text=text, ttft_s=0.1, total_s=0.2)

        understanding = {"intent": "保障", "slots": {}, "items": [], "topics": ["总保额"],
                         "gaps": [], "rewrite": "保障", "_gaps_judged": True}
        class Stream:
            def __iter__(self):
                for chunk in chunks:
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=chunk))])
                    observed_midstream.append(list(released))

            def close(self):
                pass

        with patch("agent.pipeline.cloud.chat_stream", side_effect=fake_chat_stream), \
                patch("agent.pipeline.cloud.create_with_context_retry", return_value=Stream()):
            result = Pipeline(None, client=object()).answer(
                "保障怎么解释？", understanding=understanding, voice=True,
                on_sentence=released.append)
        return result, released, observed_midstream

    def test_checked_sentence_reaches_tts_before_generation_finishes(self):
        result, released, observed = self.answer_with_chunks(["具体要看保障责任。", "还要按条款确认。"])
        self.assertEqual(observed[0], ["具体要看保障责任。"])
        self.assertEqual(result.text, "具体要看保障责任。还要按条款确认。")
        self.assertEqual(released, ["具体要看保障责任。", "还要按条款确认。"])

    def test_unsafe_sentence_never_reaches_tts(self):
        result, released, _ = self.answer_with_chunks(["保证续保。", "需要专员确认。"])
        self.assertEqual(released, [])
        self.assertTrue(result.fallback_used)

    def test_safe_prefix_survives_later_guard_failure(self):
        result, released, observed = self.answer_with_chunks(["具体要看保障责任。", "保证续保。"])
        self.assertEqual(observed[0], ["具体要看保障责任。"])
        self.assertEqual(released, ["具体要看保障责任。"])
        self.assertTrue(result.text.startswith(released[0]))
        self.assertTrue(result.fallback_used)


if __name__ == "__main__":
    unittest.main()
