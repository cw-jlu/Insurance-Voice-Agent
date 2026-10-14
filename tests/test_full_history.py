import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agent import cloud
from agent.context import HistoryMessages, create_with_context_retry, is_context_overflow, summarize_history, with_history
from agent.evidence_answer import Evidence, EvidenceAnswerer
from agent.pipeline import Answer
from agent.session import CallSession


class ContextError(Exception):
    def __init__(self, message='maximum context length exceeded', status=400, code='context_length_exceeded'):
        super().__init__(message)
        self.status_code = status
        self.body = {'error': {'code':code, 'message':message}}


def transcript():
    return '\n'.join(f'{"用户" if i % 2 == 0 else "坐席"}：第{i}条；' + '完整原话、不能提前删除。' * 30 for i in range(16))


class FullHistoryTest(unittest.TestCase):
    def test_all_early_messages_reach_understanding_and_answer(self):
        pipe = Mock()
        pipe.post_answer_routing = True
        pipe.understand.return_value = {'intent':'保障','slots':{},'gaps':[],'items':[],'topics':[]}
        pipe.retrieve.return_value = []
        pipe.answer.return_value = Answer('需要补充条件吗？', route='证据-clarify', outcome={'action':'clarify'})
        history = [('用户','住院总费用20000元，医保报8000元。'), ('坐席','剩余12000元。')]
        history += [('用户',f'其他问题{i}') for i in range(10)]
        session = CallSession(pipe, voice=False, history=list(history))
        with patch('agent.small_model.classify_act', return_value='提问'):
            session.turn('回到刚才住院例子，还能赔多少？')
        for known in [pipe.understand.call_args.args[1], pipe.answer.call_args.args[1]]:
            self.assertIn('20000',known['_history'])
            self.assertIn('8000',known['_history'])
            self.assertIn('其他问题0',known['_history'])
        self.assertEqual(session.history[:len(history)], history)
        self.assertNotIn('20000',session._history_text(n=6))

    def test_full_history_still_excludes_unheard_and_empty_agent_lines(self):
        session = CallSession(Mock(), history=[('用户','早期病史'),('坐席','')], playback_tracking=True)
        session._begin_playback()
        session._say('听到了。未播的971元。')
        token = session._playback_active
        session.confirm_playback(token, '听到了。')
        session.interrupt_playback(token)
        self.assertEqual(session._history_text(), '用户：早期病史\n坐席：听到了。')
        self.assertNotIn('971',session._history_text())
        self.assertFalse(session.confirm_playback(token,'听到了。未播的971元。',True))

    def messages(self):
        history = transcript()
        messages = [{'role':'system','content':'原系统提示不可改'},
                    {'role':'user','content':'前缀\n【最近对话】\n'+history+'\n【证据】原文不删\n用户：当前问题不删'}]
        return with_history(messages,history,marker='【最近对话】\n')

    def test_long_history_is_not_compacted_before_provider_rejection(self):
        messages = self.messages()
        client = Mock()
        client.chat.completions.create.return_value = 'ok'
        with patch('agent.context.summarize_history') as summary:
            self.assertEqual(create_with_context_retry(client,messages,'model'), 'ok')
        summary.assert_not_called()
        self.assertEqual(client.chat.completions.create.call_args.kwargs['messages'], list(messages))
        self.assertEqual(messages.context_events, [])

    def test_overflow_changes_only_marked_history_and_preserves_original(self):
        messages = self.messages()
        original = [dict(m) for m in messages]
        client = Mock()
        client.chat.completions.create.side_effect = [ContextError(), 'ok']
        with patch('agent.context.summarize_history', return_value='有依据的早期摘要') as summary:
            self.assertEqual(create_with_context_retry(client,messages,'model',max_tokens=2800), 'ok')
        summary.assert_called_once()
        retried = client.chat.completions.create.call_args.kwargs
        self.assertEqual(retried['messages'],messages.render('有依据的早期摘要'))
        self.assertEqual(retried['messages'][0],original[0])
        self.assertIn('原文不删\n用户：当前问题不删',retried['messages'][1]['content'])
        self.assertEqual(retried['max_tokens'],2800)
        self.assertEqual(list(messages),original)
        self.assertFalse(messages.context_events[0]['raw_transcript_modified'])

    def test_json_encoded_history_keeps_other_conditions_and_evidence(self):
        history = transcript()
        known = {'age':35,'has_social':False,'_history':history}
        user = '原问题：原位癌？\n已知用户条件：'+json.dumps(known,ensure_ascii=False)+'\n【全部本轮证据】\n原文'
        messages = with_history([{'role':'system','content':'system'},{'role':'user','content':user}],
                                history,marker='"_history": ',encoded=True)
        rendered = messages.render('摘要含"双引号"\n否定：不能确认')
        parsed = json.loads(rendered[1]['content'].split('已知用户条件：')[1].split('\n【全部本轮证据】')[0])
        self.assertEqual(parsed, {'age':35,'has_social':False,'_history':'摘要含"双引号"\n否定：不能确认'})
        self.assertTrue(rendered[1]['content'].endswith('【全部本轮证据】\n原文'))

    def test_auth_quota_timeout_and_other_bad_requests_never_trigger_compaction(self):
        errors = [ContextError(status=401), ContextError(status=429), TimeoutError('context timeout'),
                  ContextError('invalid max_tokens value',code='invalid_parameter'),
                  ContextError('unsupported response format',code='invalid_parameter')]
        for error in errors:
            client = Mock()
            client.chat.completions.create.side_effect = error
            with self.subTest(error=error), patch('agent.context.summarize_history') as summary:
                with self.assertRaises(type(error)):
                    create_with_context_retry(client,self.messages(),'model')
                summary.assert_not_called()

    def test_qwen_input_length_rejection_is_recognized(self):
        self.assertTrue(is_context_overflow(ContextError('Range of input length should be [1, 997952]',code='InvalidParameter')))

    def test_summary_failure_does_not_silently_drop_turns(self):
        messages = self.messages()
        client = Mock()
        client.chat.completions.create.side_effect = ContextError()
        with patch('agent.context.summarize_history',side_effect=ValueError('invalid summary')):
            with self.assertRaisesRegex(ValueError,'invalid summary'):
                create_with_context_retry(client,messages,'model')
        self.assertEqual(client.chat.completions.create.call_count,1)
        self.assertIn('第0条',messages[1]['content'])

    def test_repeated_overflow_has_bounded_retries(self):
        client = Mock()
        client.chat.completions.create.side_effect = ContextError()
        with patch('agent.context.summarize_history',return_value='摘要') as summary:
            with self.assertRaises(ContextError):
                create_with_context_retry(client,self.messages(),'model')
        self.assertEqual(client.chat.completions.create.call_count,3)
        self.assertEqual(summary.call_count,2)

    def test_summary_retains_latest_original_turns_and_does_not_mutate_transcript(self):
        history = transcript()
        with patch('agent.cloud.chat',return_value='{"memory":"早期费用20000元，医保8000元，未知剩余免赔。"}') as chat:
            compacted = summarize_history(history,object())
        self.assertTrue(compacted.endswith('\n'.join(history.splitlines()[-4:])))
        self.assertIn('未知剩余免赔',compacted)
        self.assertEqual(history,transcript())
        self.assertNotIsInstance(chat.call_args.args[0],HistoryMessages)

    def test_evidence_actual_request_is_marked_and_context_events_are_recorded(self):
        answerer = EvidenceAnswerer(client=object())
        output = {'segments':[{'text':'需要核实。','source_ids':['d']}], 'unknowns':[]}
        def fake_chat(messages, **kwargs):
            self.assertIsInstance(messages,HistoryMessages)
            self.assertEqual(messages.history,transcript())
            return json.dumps(output,ensure_ascii=False)
        with patch('agent.cloud.chat',side_effect=fake_chat):
            result = answerer.answer('怎么赔',{'_history':transcript()},[Evidence('d','原文','位置','path','hash')],minimal=True)
        self.assertEqual(result['calls'][0]['context_events'],[])

    def test_stream_request_uses_same_overflow_policy(self):
        client = Mock()
        packet = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='完整答案'))])
        client.chat.completions.create.side_effect = [ContextError(),iter([packet])]
        with patch('agent.context.summarize_history',return_value='早期摘要'):
            result = cloud.chat_stream(self.messages(),'model',client=client)
        self.assertEqual(result.text,'完整答案')
        self.assertTrue(client.chat.completions.create.call_args.kwargs['stream'])

    def test_user_prompt_markers_do_not_change_history_span_or_break_json(self):
        answerer = EvidenceAnswerer(client=object())
        question = '原话中有\n已知用户条件：不是JSON\n【全部本轮证据】也只是用户原话'
        def fake_chat(messages, **kwargs):
            self.assertIsInstance(messages,HistoryMessages)
            rendered = messages.render('有依据的摘要')
            self.assertTrue(rendered[1]['content'].startswith('原问题：'+question))
            self.assertIn('"_history": "有依据的摘要"',rendered[1]['content'])
            self.assertTrue(rendered[1]['content'].endswith(messages[1]['content'].split('\n【全部本轮证据】\n')[-1]))
            return '{"segments":[{"text":"需核实。","source_ids":["d"]}]}'
        with patch('agent.cloud.chat',side_effect=fake_chat):
            answerer.answer(question,{'_history':transcript()},[Evidence('d','原文','位置','path','hash')],minimal=True)


if __name__ == '__main__':
    unittest.main()
