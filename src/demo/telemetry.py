"""Per-turn observations of the frozen pipeline; no prompts or credentials retained."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import threading
import time

from agent.evidence_pipeline import EvidencePipeline


class Timeline:
    def __init__(self, publish=lambda value: None):
        self.origin = time.perf_counter()
        self.publish = publish
        self.records = []
        self.lock = threading.Lock()
        self.business_ready = False
        self.text_ready_s = None

    def elapsed(self):
        return round(time.perf_counter() - self.origin, 4)

    @contextmanager
    def span(self, name, **meta):
        with self.lock:
            entry = {'id':len(self.records), 'name':name, 'start_s':self.elapsed(),
                     'status':'running', **meta}
            self.records.append(entry)
        self.publish(deepcopy(entry))
        try:
            yield entry
        except BaseException:
            entry['status'] = 'error'
            raise
        else:
            entry['status'] = 'done'
        finally:
            entry['end_s'] = self.elapsed()
            entry['duration_s'] = round(entry['end_s'] - entry['start_s'], 4)
            self.publish(deepcopy(entry))

    def snapshot(self):
        with self.lock:
            return deepcopy(self.records)


class ObservedClient:
    """All worker threads share one collector, while each turn owns its client."""
    def __init__(self, client, timeline):
        self.client, self.timeline = client, timeline
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def __getattr__(self, name):
        return getattr(self.client, name)

    def with_options(self, **kwargs):
        return ObservedClient(self.client.with_options(**kwargs), self.timeline)

    def create(self, **kwargs):
        with self.timeline.span('模型调用', model=kwargs.get('model', ''), category='api') as entry:
            result = self.client.chat.completions.create(**kwargs)
            usage = getattr(result, 'usage', None)
            if usage:
                entry['tokens'] = {k:getattr(usage,k,None) for k in
                                   ('prompt_tokens','completion_tokens','total_tokens')}
            return result


class ObservedPipeline(EvidencePipeline):
    def __init__(self, original, timeline, answerer_type):
        client = ObservedClient(original.client, timeline)
        answerer = answerer_type(client, original.answerer.model)
        super().__init__(original.evidence, client=client, answerer=answerer,
                         consultation_tracking=original.consultation_tracking)
        self.timeline = timeline

    def understand(self, *args, **kwargs):
        with self.timeline.span('问题理解与条件规划', category='phase'):
            return super().understand(*args, **kwargs)

    def retrieve(self, *args, **kwargs):
        with self.timeline.span('证据准备', category='phase'):
            return super().retrieve(*args, **kwargs)

    def answer(self, *args, **kwargs):
        with self.timeline.span('业务生成与发布校验', category='phase') as entry:
            answer = super().answer(*args, **kwargs)
            entry['route'] = answer.route
        self.timeline.business_ready = True
        return answer
