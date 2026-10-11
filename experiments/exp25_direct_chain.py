"""Live full text-chain comparison, original user input, legacy vs direct DS."""
import json
import sys
import statistics
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments')]
from agent import cloud
from agent.pipeline import Pipeline, HybridRetriever
from agent.session import CallSession
from embed_cache import corpus, load
from data_complex import items

def main():
    client = cloud.llm_client()
    chunks = corpus(False)
    retriever = HybridRetriever(chunks, load(chunks, 'clauses'), client=client)
    pipes = {name: Pipeline(retriever, client=client, direct_answer=name == 'direct') for name in ('legacy', 'direct')}
    cases = [items()[i] for i in (1, 3, 6, 10, 18, 26)] + [
        {'id': 'S01', 'question': '药省保每月最多报多少'},
        {'id': 'S02', 'question': '特定药品要在哪里购买'}]
    result = {'model': cloud.SLOW_MODEL, 'rounds': 2, 'n_cases': len(cases), 'rows': [], 'calls': []}
    path = ROOT / 'experiments/results/exp25_direct_chain_v2.json'
    old_path = ROOT / 'experiments/results/exp25_direct_chain.json'
    if path.exists():
        result = json.loads(path.read_text(encoding='utf-8'))
    elif old_path.exists():
        previous = json.loads(old_path.read_text(encoding='utf-8'))
        result['rows'] = [r for r in previous['rows'] if r['variant'] == 'legacy']
    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    original_stream = cloud.chat_stream
    for round_ in (1, 2):
        for variant in (('legacy', 'direct') if round_ == 1 else ('direct', 'legacy')):
            pipe = pipes[variant]
            for case in cases:
                if any(r['id'] == case['id'] and r['round'] == round_ and r['variant'] == variant for r in result['rows']):
                    continue
                messages = []
                def capture(msgs, *args, **kwargs):
                    if kwargs.get('model') == pipe.slow_model:
                        messages.append(msgs)
                    return original_stream(msgs, *args, **kwargs)
                try:
                    with patch('agent.pipeline.cloud.chat_stream', side_effect=capture):
                        a = pipe.answer(case['question'], voice=True)
                    row = {'id': case['id'], 'round': round_, 'variant': variant, 'question': case['question'],
                           'text': a.text, 'chars': len(a.text), 'timings': a.timings, 'route': a.route,
                           'fallback': a.fallback_used, 'violations': a.violations, 'attempts': a.attempts,
                           'understanding': {k: v for k, v in a.understanding.items() if k != '_retrieved'},
                           'answer_messages': messages}
                except Exception as exc:
                    row = {'id': case['id'], 'round': round_, 'variant': variant, 'error_type': type(exc).__name__}
                result['rows'].append(row)
                save()
                print(json.dumps({k: v for k,v in row.items() if k not in ('timings','answer_messages','understanding')}, ensure_ascii=False), flush=True)
    for question in ('我30岁有社保，年缴和月缴差多少', '你们天天打电话太烦了！', '不要再打电话了'):
        s = CallSession(pipes['direct'])
        s.opening()
        r = s.turn(question)
        result['calls'].append({'question': question, 'text': r.text, 'events': r.events, 'timings': r.timings,
                                'profile': s.profile, 'handoff': s.flags['handoff'], 'ended': s.ended})
    for variant in ('legacy', 'direct'):
        rows = [r for r in result['rows'] if r['variant'] == variant and 'text' in r]
        print(variant, 'n', len(rows), 'chars', statistics.median(r['chars'] for r in rows),
              'total_s', statistics.median(r['timings']['total'] for r in rows), 'fallbacks', sum(r['fallback'] for r in rows), flush=True)
    save()

if __name__ == '__main__':
    main()
