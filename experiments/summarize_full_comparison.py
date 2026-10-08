"""Summarize full comparison without treating model grading as human accuracy."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results'


def stats(values):
    xs = sorted(x for x in values if x is not None)
    return {'n': len(xs), 'p50': round(statistics.median(xs), 3),
            'p90': round(xs[int(.9 * (len(xs)-1))], 3), 'max': round(max(xs), 3)} if xs else {'n': 0}


def main():
    result = {}
    for provider in ('qwen', 'deepseek'):
        summary = result[provider] = {}
        for mode in ('qa', 'calls', 'voice'):
            path = OUT / f'exp21_{provider}_{mode}.json'
            if not path.exists():
                continue
            d = json.loads(path.read_text(encoding='utf-8'))
            all_rows = d['rows']
            rows = [r for r in all_rows if 'error' not in r]
            block = summary[mode] = {'completed': len(all_rows), 'valid': len(rows),
                                     'errors': len(all_rows)-len(rows), 'planned': d.get('planned', 72)}
            if mode == 'qa':
                groups = defaultdict(list)
                for r in rows:
                    groups[r['dataset']].append(r)
                def quality(rs):
                    judged = [r for r in rs if r['judge_ok'] is not None]
                    return {'n': len(rs), 'judged': len(judged),
                            'regex_pass': sum(r['regex_ok'] for r in rs),
                            'judge_pass': sum(r['judge_ok'] for r in judged),
                            'both_pass': sum(r['regex_ok'] and r['judge_ok'] for r in judged),
                            'fallback': sum(r['fallback'] for r in rs),
                            'answer_s': stats([r['answer_s'] for r in rs])}
                block.update(quality(rows))
                block['datasets'] = {k: quality(v) for k,v in groups.items()}
            elif mode == 'calls':
                checks = defaultdict(lambda: [0,0])
                for r in rows:
                    for k,v in r['evaluation']['checks'].items():
                        if v is not None:
                            checks[k][0] += bool(v)
                            checks[k][1] += 1
                block['checks'] = dict(checks)
                block['calls_with_problem'] = sum(bool(r['evaluation']['problems']) for r in rows)
                block['calls_with_failed_check'] = sum(any(v is False for v in r['evaluation']['checks'].values()) for r in rows)
                block['outcomes'] = dict((k, sum(r['evaluation']['outcome']==k for r in rows))
                                         for k in sorted({r['evaluation']['outcome'] for r in rows}))
            else:
                groups = defaultdict(list)
                for r in rows:
                    groups[r['arm']].append(r)
                block['arms'] = {}
                for arm, rs in groups.items():
                    generated = [r for r in rs if r['timings'].get('answer_detail', {}).get('generate_1')]
                    block['arms'][arm] = {'n': len(rs), 'reuse': sum(r['preparation']['reused'] for r in rs),
                        'opener_first_response_s': stats([r['first_response_s'] for r in rs]),
                        'first_audio_s': stats([r['first_substantive_audio_s'] for r in rs]),
                        'simulated_playback_s': stats([r['simulated_playback_end_s'] for r in rs]),
                        'generated_n': len(generated),
                        'generated_first_audio_s': stats([r['first_substantive_audio_s'] for r in generated]),
                        'generated_simulated_playback_s': stats([r['simulated_playback_end_s'] for r in generated]),
                        'generated_over_10s': sum(r['simulated_playback_end_s']>10 for r in generated)}
        endpoint = OUT / f'exp21_{provider}_endpoint.json'
        if endpoint.exists():
            rows = json.loads(endpoint.read_text(encoding='utf-8'))['rows']
            summary['endpoint'] = {}
            for vad in (300,500):
                rs = [r for r in rows if r['vad_ms']==vad]
                summary['endpoint'][str(vad)] = {'n': len(rs),
                    'split_turns': sum(bool(r.get('error_reason')) for r in rs),
                    'service_errors': sum('error' in r and not r.get('error_reason') for r in rs),
                    'asr_endpoint_s': stats([r.get('asr_endpoint_s') for r in rs])}
    interrupt = json.loads((OUT / 'exp21_interrupt.json').read_text(encoding='utf-8'))
    result['shared_interrupt'] = interrupt['summary']
    for cond, block in result['shared_interrupt'].items():
        rs = [r for r in interrupt['rows'] if r.get('condition')==cond and r.get('kind')=='打断']
        block['stop_within_500ms'] = sum(r.get('stop_latency_s') is not None and 0<=r['stop_latency_s']<=.5 for r in rs)
        block['stopped_before_user_onset'] = sum(r.get('stop_latency_s') is not None and r['stop_latency_s']<0 for r in rs)
    (OUT / 'exp21_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
