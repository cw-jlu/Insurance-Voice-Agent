"""Provider switching must preserve fast-task routing and timeout options."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from model_router import AnswerModelRouter


class RouterTest(unittest.TestCase):
    def test_default_cloud_client_routes_production_answer(self):
        from agent import cloud
        qwen = Mock()
        with patch.object(cloud, 'ANSWER_PROVIDER', 'deepseek'), patch.object(cloud, 'SLOW_MODEL', 'deepseek-flash'), patch.object(cloud, 'qwen_client', return_value=qwen), patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test-only'}), patch('openai.OpenAI') as constructor:
            client = cloud.llm_client()
            client.chat.completions.create(model=cloud.SLOW_MODEL, messages=[])
            constructor.return_value.chat.completions.create.assert_called_once()
            client.chat.completions.create(model=cloud.FAST_MODEL, messages=[])
            qwen.chat.completions.create.assert_called_once()
            self.assertIs(client.embeddings, qwen.embeddings)

    def test_qwen_configuration_uses_qwen_client(self):
        from agent import cloud
        qwen = Mock()
        with patch.object(cloud, 'ANSWER_PROVIDER', 'qwen'), patch.object(cloud, 'qwen_client', return_value=qwen):
            self.assertIs(cloud.llm_client(), qwen)

    def router(self):
        router = object.__new__(AnswerModelRouter)
        router.qwen, router.deepseek = Mock(), Mock()
        router.model = "deepseek-flash"
        return router

    def test_only_answer_model_uses_deepseek_and_its_thinking_option(self):
        router = self.router()
        router.create(model="qwen-flash", messages=[])
        router.qwen.chat.completions.create.assert_called_once()
        router.deepseek.chat.completions.create.assert_not_called()
        router.create(model="deepseek-flash", messages=[], stream=True, extra_body={"enable_thinking": False})
        kwargs = router.deepseek.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertTrue(kwargs["stream"])

    def test_understanding_timeout_copy_preserves_provider_routing(self):
        router = self.router()
        copy = router.with_options(timeout=3, max_retries=1)
        copy.chat.completions.create(model="qwen-flash", messages=[])
        router.qwen.with_options.assert_called_once_with(timeout=3, max_retries=1)
        router.qwen.with_options.return_value.chat.completions.create.assert_called_once()
        router.deepseek.chat.completions.create.assert_not_called()
