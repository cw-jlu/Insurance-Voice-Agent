"""Small live text-chain regression; no STT or TTS calls."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments')]
from agent.pipeline import Pipeline, HybridRetriever
from agent import cloud
from agent.session import CallSession
from embed_cache import corpus, load
from data_complex import items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--extra', action='store_true')
    args = ap.parse_args()
    chunks = corpus(False)
    pipe = Pipeline(HybridRetriever(chunks, load(chunks, 'clauses')))
    rows = []
    for index in ((11, 26) if args.extra else (0, 2, 8, 9, 17, 25)):
        case = items()[index]
        question = case.get('q', case.get('question'))
        try:
            answer = pipe.answer(question, voice=True)
        except Exception as exc:
            rows.append({'id': case['id'], 'error_type': type(exc).__name__, 'status_code': getattr(exc, 'status_code', None)})
            (ROOT / 'experiments/results' / f'exp23_{args.label}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
            raise SystemExit(type(exc).__name__) from None
        row = {'id': case.get('id'), 'question': question, 'text': answer.text,
               'route': answer.route, 'timings': answer.timings, 'attempts': answer.attempts,
               'violations': answer.violations, 'fallback': answer.fallback_used,
               'understanding': {k: v for k, v in answer.understanding.items() if k != '_retrieved'},
               'chars': len(answer.text), 'model': pipe.slow_model}
        rows.append(row)
        path = ROOT / 'experiments/results' / f'exp23_{args.label}.json'
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if args.extra:
        for question in ('你们天天打电话太烦了！', '你们天天打电话太烦了，不要再打电话'):
            session = CallSession(pipe)
            session.opening()
            result = session.turn(question)
            rows.append({'question': question, 'text': result.text, 'events': result.events,
                         'timings': result.timings, 'handoff': session.flags['handoff'], 'ended': session.ended})
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
