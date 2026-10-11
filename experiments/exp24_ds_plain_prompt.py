"""Prompt-only paired generation from saved identical evidence, no slot assembly/fallback."""
import json
import statistics
from pathlib import Path
from exp24_ds_minimum_prompt import ROOT, MINIMUM_SYS
from agent import cloud, pipeline
from agent.pipeline import grouped_context, item_chunks, derived_numbers
from agent.knowledge import clause_chunks
from agent.guardrail import extract_numbers

def main():
    previous = json.loads((ROOT / 'experiments/results/exp24_ds_minimum_prompt.json').read_text(encoding='utf-8'))
    lookup = {c.chunk_id: c for c in clause_chunks()}
    for iid in pipeline.COVERAGE:
        lookup.update({c.chunk_id: c for c in item_chunks(iid)})
    lookup['family'] = pipeline.Chunk('family', '费率表', pipeline.FAMILY_RULE, '费率表 家庭单用户优享价格规则说明')
    rows = [r for r in previous['rows'] if r['variant'] == 'current' and r['round'] == 1]
    client = cloud.llm_client()
    result = {'model': cloud.SLOW_MODEL, 'rounds': 2, 'n_cases': len(rows),
              'mode': 'plain generation, identical question/evidence; guards recorded without replacing answer', 'rows': []}
    path = ROOT / 'experiments/results/exp24_ds_plain_prompt.json'
    for round_ in (1, 2):
        for variant in (('current', 'minimum') if round_ == 1 else ('minimum', 'current')):
            for old in rows:
                ids = next(t['evidence'] for t in old['timings']['trace'] if t['stage'] == 'resolve')
                evidence = [lookup[i] for i in ids]
                question = old['question']
                ctx = grouped_context(evidence)
                prompt = f'【资料】\n{ctx}\n\n用户：{question}\n只输出给用户听的回答。'
                try:
                    response = cloud.chat_stream([{'role': 'system', 'content': previous['prompts']['current'] if variant == 'current' else MINIMUM_SYS},
                                                  {'role': 'user', 'content': prompt}], model=cloud.SLOW_MODEL, client=client, temperature=.3)
                    structured = {n.value for n in extract_numbers(question)} | derived_numbers(question, evidence)
                    guard = pipeline._guard(question, response.text, evidence, structured, [], gaps_judged=True,
                                            fact_check='off', features=frozenset({'D'}), client=client)
                    row = {'id': old['id'], 'round': round_, 'variant': variant, 'question': question,
                           'text': response.text, 'chars': len(response.text), 'generate_s': response.total_s,
                           'ttft_s': response.ttft_s, 'guard_violations': guard, 'evidence': ids}
                except Exception as exc:
                    row = {'id': old['id'], 'round': round_, 'variant': variant, 'error_type': type(exc).__name__}
                result['rows'].append(row)
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps(row, ensure_ascii=False), flush=True)
    for variant in ('current', 'minimum'):
        selected = [r for r in result['rows'] if r['variant'] == variant and 'text' in r]
        print(variant, len(selected), statistics.median(r['chars'] for r in selected),
              statistics.median(r['generate_s'] for r in selected), flush=True)

if __name__ == '__main__':
    main()
