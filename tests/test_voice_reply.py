"""Regression checks for concise voice replies and the pre-link disclosure gate."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.cloud import StreamResult  # noqa: E402
from agent.pipeline import Answer, Pipeline, WAIT_ADDON_TEXT, WAIT_TEXT, addon_notice, assemble_voice_slots, item_chunks, understand_merged  # noqa: E402
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
    def test_stream_does_not_release_half_sentence_before_late_bad_number(self):
        prefix = "这项责任需要符合合同规定的条件，" * 6
        closed = []
        class Stream:
            def __iter__(self):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=prefix))])
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="赔付999999999元。"))])
            def close(self):
                closed.append(True)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: Stream())))
        spoken = []
        _, violations = Pipeline(None, client=client)._stream_generate([], [], set(), spoken.append, {})
        self.assertEqual(spoken, [])
        self.assertTrue(violations)
        self.assertEqual(closed, [True])

    def test_stream_preserves_separate_purchase_notice(self):
        class Stream:
            def __iter__(self):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ICU津贴按合同给付。"))])
            def close(self):
                pass
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: Stream())))
        spoken = []
        Pipeline(None, client=client)._stream_generate([], item_chunks("R15"), set(), spoken.append, {})
        self.assertIn("另外选购", "".join(spoken))

    def test_session_delivers_sentence_before_pipeline_returns(self):
        spoken = []
        class StreamingPipeline(FakePipeline):
            def answer(self, _question, _known, **kwargs):
                callback = kwargs.get("on_sentence")
                if callback:
                    callback("第一项能报。")
                    # Check delivery inside generation, before returning the complete answer.
                    assert spoken == ["我帮您查一下。", "第一项能报。"]
                    callback("第二项需另购。")
                return Answer("第一项能报。第二项需另购。", route="保障", timings={"total": 0.0})
        session = self.session("")
        session.pipe = StreamingPipeline("")
        result = session.turn("两个保障分别怎么报？", on_sentence=spoken.append)
        self.assertEqual(result.text, "".join(spoken))
        self.assertEqual(spoken, ["我帮您查一下。", "第一项能报。", "第二项需另购。"])

    def setUp(self):
        self.act = patch("agent.session.small_model.classify_act",
                         side_effect=lambda text, *_: "提问" if "？" in text else "附和")
        self.gaps = patch("agent.session.small_model.detect_gaps", return_value=[])
        self.act.start()
        self.gaps.start()
        self.addCleanup(self.act.stop)
        self.addCleanup(self.gaps.stop)

    def session(self, reply: str) -> CallSession:
        s = CallSession(FakePipeline(reply), playback_tracking=False)  # Text delivery fixture, no device.
        s.opening()
        s.profile.update({"relation": "本人", "age": 30, "has_social": True})
        s.flags["quoted"] = True
        return s

    def test_answer_is_not_cut_or_followed_by_generic_limits(self):
        reply = "住院按条款报销。门诊手术按对应责任报销。" + "".join(
            f"第{i}项条件依保单约定。" for i in range(1, 8))
        s = self.session(reply)
        result = s.turn("住院和门诊手术都能报吗？")
        self.assertEqual(result.text, '我帮您查一下。' + reply)
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

    def test_experimental_streaming_does_not_hard_cut_long_reply(self):
        sentences = ["这项保障按照对应条款约定处理，请核对适用责任和必要条件。" * 2 + "。",
                     "另一个问题也应完整回答，不能因口播长度省略结论。"]

        class FakeStream:
            def __iter__(self):
                for sentence in sentences:
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=sentence))])

            def close(self):
                pass

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: FakeStream())))
        pipe = Pipeline(None, client=client)
        understanding = {"intent": "保障", "slots": {}, "items": [], "topics": [], "gaps": [], "_retrieved": []}
        spoken = []
        reply = pipe.answer("两个问题分别说明？", voice=True, understanding=understanding, on_sentence=spoken.append)
        self.assertEqual(reply.text, "".join(spoken))
        self.assertIn("另一个问题", reply.text)
        self.assertGreater(len(reply.text), 50)

    def test_merged_understanding_formats_current_prompt(self):
        raw = '{"intent":"保障","slots":{},"items":[],"topics":[],"gaps":[],"act":"提问","gaps_now":[]}'
        with patch("agent.pipeline.cloud.chat", return_value=raw) as chat:
            result = understand_merged("住院能报吗？", {}, client=object())
        self.assertEqual(result["act"], "提问")
        self.assertIn("最多 5 个", chat.call_args.args[0][0]["content"])


if __name__ == "__main__":
    unittest.main()
