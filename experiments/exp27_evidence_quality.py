"""Frozen 71-question evidence/prompt/blind-deliberation comparison, no answer repairs."""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from agent import cloud
from agent.pipeline import Pipeline, HybridRetriever
from agent.evidence_answer import EvidenceAnswerer, full_evidence, sparse_evidence, SYSTEM, INDEPENDENT, CHALLENGE, ARBITER, MINIMAL_SYSTEM, BOUND_SYSTEM
from embed_cache import corpus,load
from exp18_voice_prompt_ablation import rows_for
from exp2_qa_v2 import regex_judge
from dataclasses import asdict


def save(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variants',default='prompt_sparse,evidence_full,debate_full')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--ids',default='')
    parser.add_argument('--output',default='exp27_evidence_quality_v1.json')
    args=parser.parse_args()
    variants=args.variants.split(',')
    if set(variants)-{'prompt_sparse','evidence_full','debate_full','legacy','minimal_full','bound_full'}:
        parser.error('unknown variant')
    cases=rows_for('all')
    if args.ids:
        cases=[r for r in cases if r['id'] in args.ids.split(',')]
    client=cloud.llm_client()
    full=full_evidence(labelled=variants==['bound_full'])
    snapshot=args.output.replace('.json','_sources.json')
    save(ROOT/'experiments/results'/snapshot,{'evidence':[asdict(e)for e in full]})
    answerer=EvidenceAnswerer(client)
    chunks=corpus(False)
    direct=Pipeline(HybridRetriever(chunks,load(chunks,'clauses'),client=client),client=client,direct_answer=True)
    legacy=Pipeline(direct.retriever,client=client,direct_answer=False)
    path=ROOT/'experiments/results'/args.output
    result=json.loads(path.read_text(encoding='utf-8'))if path.exists()else{
        'model':cloud.SLOW_MODEL,'independent_model':cloud.QWEN_ANSWER_MODEL,
        'prompts':{'single':SYSTEM,'independent':INDEPENDENT,'challenge':CHALLENGE,'arbiter':ARBITER,'minimal':MINIMAL_SYSTEM,'bound':BOUND_SYSTEM},
        'snapshot':snapshot,
        'design':'71 frozen cases; no reference facts passed to generation; no answer repair; full identical pack for evidence_full/debate_full',
        'rows':[]}
    done={(r['id'],r['source'],r['variant'])for r in result['rows']if 'error_type'not in r}
    def one(case,variant):
        row={'id':case['id'],'source':case['source'],'variant':variant,'question':case['question']}
        try:
            if variant=='legacy':
                a=legacy.answer(case['question'],dict(case.get('context')or{}),voice=True)
                row.update(text=a.text,total_s=a.timings.get('total'),timings=a.timings,
                           evidence_ids=[c.chunk_id for c in a.evidence],route=a.route,fallback=a.fallback_used)
            else:
                ev=sparse_evidence(direct.retrieve(case['question']),full)if variant=='prompt_sparse'else full
                row.update(answerer.answer(case['question'],dict(case.get('context')or{}),ev,debate=variant=='debate_full',minimal=variant=='minimal_full',bound=variant=='bound_full'))
            row['regex_ok'],row['regex_missing'],row['regex_forbidden']=regex_judge(case,row['text'])
        except Exception as exc:
            row['error_type']=type(exc).__name__
        return row
    with ThreadPoolExecutor(args.workers)as pool:
        jobs=[pool.submit(one,c,v)for c in cases for v in variants if(c['id'],c['source'],v)not in done]
        for f in as_completed(jobs):
            row=f.result()
            result['rows']=[r for r in result['rows']if(r['id'],r['source'],r['variant'])!=(row['id'],row['source'],row['variant'])]
            result['rows'].append(row)
            save(path,result)
            print(json.dumps({k:row.get(k)for k in('id','variant','total_s','provenance','regex_ok','error_type')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
