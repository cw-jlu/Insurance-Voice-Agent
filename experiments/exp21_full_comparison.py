"""Full existing QA sets and complete call personas; incremental, secret-free results."""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments')]
from agent import cloud
from agent.pipeline import Pipeline, HybridRetriever
from embed_cache import corpus, load
from model_router import AnswerModelRouter
from data_complex import items as complex_items
import exp2_qa_v2 as qa
import exp8_calls as calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', choices=['qwen', 'deepseek'], required=True)
    ap.add_argument('--mode', choices=['qa', 'calls'], required=True)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    load_dotenv(ROOT / '.env')
    load_dotenv(ROOT / '.env.deepseek')
    base = cloud.qwen_client().with_options(timeout=30, max_retries=1)
    model = cloud.QWEN_ANSWER_MODEL if args.provider == 'qwen' else 'deepseek-flash'
    client = base if args.provider == 'qwen' else AnswerModelRouter(base, model)
    chunks = corpus(False)
    pipe = Pipeline(HybridRetriever(chunks, load(chunks, 'clauses'), client=base),
                    client=client, slow_model=model)
    # Same Qwen Plus judge/simulator for both answer models; automated grading is provisional.
    qa.JUDGE_MODEL = cloud.QWEN_ANSWER_MODEL
    calls.SIM_MODEL = cloud.QWEN_ANSWER_MODEL
    path = ROOT / 'experiments/results' / f'exp21_{args.provider}_{args.mode}.json'
    if args.mode == 'qa':
        jobs = [{**r, 'dataset': 'complex'} for r in complex_items()]
        for file in sorted((ROOT / 'experiments/data').glob('qa_v*.json')):
            jobs += [{**r, 'dataset': file.stem} for r in json.loads(file.read_text(encoding='utf-8'))
                     if r.get('valid') is not False]
    else:
        jobs = [{'persona': p, 'round': i + 1} for i in range(args.rounds)
                for p in calls.PERSONAS + calls.PERSONAS_V2 + calls.PERSONAS_V3]

    def one(job):
        if args.mode == 'calls':
            result = calls.simulate(job['persona'], pipe, base)
            return {'round': job['round'], 'call': result,
                    'evaluation': calls.evaluate(result, job['persona'], base)}
        known = dict(job.get('context') or {})
        if job.get('turns'):
            for utterance in job['turns'][:-1]:
                u = pipe.understand(utterance, known)
                known.update({k: v for k, v in (u.get('slots') or {}).items() if v is not None})
                known['_prev_utterance'], known['_prev_intent'] = utterance, u.get('intent')
        t0 = time.perf_counter()
        answer = pipe.answer(job['question'], known, voice=True)
        elapsed = time.perf_counter() - t0
        ok, missing, forbidden = qa.regex_judge(job, answer.text)
        try:
            judged, reason = qa.model_judge(job, answer.text, base)
        except Exception as exc:
            judged, reason = None, type(exc).__name__
        return {'id': job['id'], 'dataset': job['dataset'], 'question': job['question'],
                'facts': job['facts'], 'text': answer.text, 'regex_ok': ok, 'missing': missing,
                'forbidden': forbidden, 'judge_ok': judged, 'judge_reason': reason,
                'answer_s': elapsed, 'fallback': answer.fallback_used,
                'violations': answer.violations, 'route': answer.route}

    def key(job):
        return ((job['dataset'], job['id']) if args.mode == 'qa'
                else (job['persona']['name'], job['round']))

    rows = []
    planned = len(jobs)
    if args.resume and path.exists():
        rows = [r for r in json.loads(path.read_text(encoding='utf-8'))['rows'] if 'error' not in r]
        done = {(r['dataset'], r['id']) if args.mode == 'qa'
                else (r['call']['persona'], r['round']) for r in rows}
        jobs = [j for j in jobs if key(j) not in done]
    metadata = {'provider': args.provider, 'answer_model': model, 'judge_model': cloud.QWEN_ANSWER_MODEL,
                'mode': args.mode, 'planned': planned, 'voice': True,
                'quality': 'automated grading; requires human review', 'rows': rows}
    with ThreadPoolExecutor(3) as pool:
        futures = {pool.submit(one, job): job for job in jobs}
        for f in as_completed(futures):
            job = futures[f]
            try:
                row = f.result()
            except Exception as exc:
                row = {'id': job.get('id'), 'dataset': job.get('dataset'),
                       'persona': (job.get('persona') or {}).get('name'), 'round': job.get('round'),
                       'error': type(exc).__name__}
            rows.append(row)
            payload = json.dumps(metadata, ensure_ascii=False, indent=2)
            for attempt in range(10):
                try:
                    temp = path.with_suffix('.tmp')
                    temp.write_text(payload, encoding='utf-8')
                    temp.replace(path)
                    break
                except OSError:
                    if attempt == 9:
                        raise
                    time.sleep(.1)
            if len(rows) % 10 == 0 or len(rows) == planned:
                print(json.dumps({'done': len(rows), 'total': planned,
                                  'errors': sum('error' in r for r in rows)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
