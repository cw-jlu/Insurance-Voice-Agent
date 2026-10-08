"""Regression checks for concise voice replies and the pre-link disclosure gate."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.cloud import StreamResult  # noqa: E402
from agent.pipeline import Answer, Pipeline, WAIT_ADDON_TEXT, WAIT_TEXT, addon_notice, assemble_voice_slots, item_chunks  # noqa: E402
from agent.session import CallSession, DISCLOSURE, LIMITS  # noqa: E402


class FakePipeline:
    client = None

    def __init__(self, reply: str):
        self.reply = reply

    def understand(self, _question, _known):
        return {"intent": "保障", "slots": {}, "items": [], "topics": [], "gaps": []}

    def retrieve(self, _question):
        return []

    def answer(self, _question, _known, **_kwargs):
        return Answer(self.reply, route="保障", timings={"total": 0.0})


class VoiceReplyTest(unittest.TestCase):
    def setUp(self):
        self.act = patch("agent.session.small_model.classify_act",
                         side_effect=lambda text, *_: "提问" if "？" in text else "附和")
        self.gaps = patch("agent.session.small_model.detect_gaps", return_value=[])
        self.act.start()
        self.gaps.start()
        self.addCleanup(self.act.stop)
        self.addCleanup(self.gaps.stop)

    def session(self, reply: str) -> CallSession:
        s = CallSession(FakePipeline(reply))
        s.opening()
        s.profile.update({"relation": "本人", "age": 30, "has_social": True})
        s.flags["quoted"] = True
        return s

    def test_answer_is_not_cut_or_followed_by_generic_limits(self):
        reply = "住院按条款报销。门诊手术按对应责任报销。" + "".join(
            f"第{i}项条件依保单约定。" for i in range(1, 8))
        s = self.session(reply)
        result = s.turn("住院和门诊手术都能报吗？")
        self.assertEqual(result.text, reply)
        self.assertNotIn("提示限制", result.events)
        self.assertFalse(s.flags["limits"])

        next_turn = s.turn("好的")
        self.assertEqual(next_turn.text, LIMITS)
        self.assertTrue(s.flags["limits"])

    def test_direct_link_request_still_requires_limits_and_disclosure(self):
        s = self.session("住院按条款报销。")
        s.turn("住院能报吗？")
        result = s.turn("把投保链接发我手机")
        self.assertFalse(s.flags["link"])
        self.assertIn(LIMITS, result.text)
        self.assertIn(DISCLOSURE, result.text)
        self.assertTrue(s.flags["limits"])
        self.assertTrue(s.flags["disclosure"])

    def test_pipeline_does_not_drop_model_sentences_at_a_character_limit(self):
        reply = "".join(f"第{i}项按对应条款办理。" for i in range(1, 15))
        understanding = {"intent": "保障", "slots": {}, "items": [], "topics": [], "gaps": [], "_retrieved": []}
        pipe = Pipeline(None, client=object())
        with patch("agent.pipeline.cloud.chat_stream",
                   return_value=StreamResult(reply, 0.0, 0.0)), patch("agent.pipeline._guard", return_value=[]):
            answer = pipe.answer("这些项目分别怎么办？", voice=True, understanding=understanding)
        self.assertEqual(answer.text, reply)

    def test_waiting_period_template_only_mentions_the_asked_product(self):
        understanding = {"intent": "保障", "slots": {}, "items": [], "topics": [], "gaps": [], "_retrieved": []}
        pipe = Pipeline(None, client=object())
        self.assertEqual(pipe.answer("等待期多久？", voice=True, understanding=understanding).text, WAIT_TEXT)
        self.assertEqual(pipe.answer("药省保加油包等待期多久？", voice=True,
                                     understanding=understanding).text, WAIT_ADDON_TEXT)

    def test_addon_notice_does_not_repeat_other_words_for_separate_purchase(self):
        reply = "住院津贴每天100元，需另购尊享住院津贴加油包。"
        self.assertEqual(addon_notice(reply, item_chunks("R14")), reply)

    def test_addon_retrieval_does_not_attach_notice_to_base_coverage(self):
        reply = "一般门急诊和住院共享1万元免赔额，扣除后按约定比例赔。"
        self.assertEqual(addon_notice(reply, item_chunks("R12") + item_chunks("R13")), reply)
        self.assertIn("另外选购", addon_notice("ICU津贴每天500元。", item_chunks("R15")))

    def test_voice_slots_require_every_question_and_preserve_all_fields(self):
        raw = ('{"answers":[{"id":1,"conclusion":"住院能报","numbers":"扣1万免赔额",'
               '"condition":"按条款比例"},{"id":2,"conclusion":"ICU津贴需另购",'
               '"numbers":"每天500元","condition":""}]}')
        self.assertEqual(assemble_voice_slots(raw, 2),
                         "住院能报，扣1万免赔额，按条款比例。ICU津贴需另购，每天500元。")
        self.assertEqual(assemble_voice_slots(raw, 3), "")
        self.assertEqual(assemble_voice_slots('{"answers":[{"id":2,"conclusion":"跳项"}]}', 1), "")


if __name__ == "__main__":
    unittest.main()
