"""Send the answer model to DeepSeek, retaining Qwen fast tasks and embeddings."""
import os
from types import SimpleNamespace


class AnswerModelRouter:
    def __init__(self, qwen, answer_model):
        from openai import OpenAI
        if not os.getenv('DEEPSEEK_API_KEY'):
            raise ValueError('Missing service configuration: DEEPSEEK_API_KEY')
        self.qwen = qwen
        self.model = answer_model
        self.deepseek = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'],
                               base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
                               timeout=30, max_retries=0)
        self.embeddings = qwen.embeddings
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **options):
        router = object.__new__(type(self))
        router.qwen = self.qwen.with_options(**options)
        router.deepseek = self.deepseek.with_options(**options)
        router.model = self.model
        router.embeddings = router.qwen.embeddings
        router.chat = SimpleNamespace(completions=SimpleNamespace(create=router.create))
        return router

    def create(self, **kwargs):
        if kwargs.get('model') != self.model:
            return self.qwen.chat.completions.create(**kwargs)
        extra = dict(kwargs.pop('extra_body', {}) or {})
        extra.pop('enable_thinking', None)
        extra['thinking'] = {'type': 'disabled'}
        return self.deepseek.chat.completions.create(**kwargs, extra_body=extra)
