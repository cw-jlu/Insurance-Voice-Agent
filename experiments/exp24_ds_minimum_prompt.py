"""Paired live DS prompt experiment, shared understanding and evidence; no speech."""
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments')]
from agent import cloud, pipeline
from agent.pipeline import Pipeline, HybridRetriever
from embed_cache import corpus, load
from data_complex import items
from exp18_voice_prompt_ablation import prepare

MINIMUM_SYS = '''你是众安保险AI电话助理，回答尊享e生百万医疗2026版问题，只依据本轮资料与程序报价，不编造。
在内部先选择本轮必须说的事实，再组织最终短答，不输出思考过程。
选事实的唯一标准：直接回答用户实际问到的一件事，或者省略后会让这个答案错误或误导。背景条件不是新的问题。
用户没问的年度保额、医院范围、等待期、流程、天数上限和其他产品，不主动介绍。必要的免责、另购要求、核保边界不能省。
同一条件只讲一次，多个问题用短句并列；不复述问题，不解释计算过程，不写铺垫、总结、销售引导或反问。答完即停，细节留给追问。
责任的比例、免赔、等待期不能跨责任套用；有社保身份投保却不用社保结算与无社保身份投保不同。既往症免责与核保结果分开说。
资料不足时，只对未知部分说明无法确认，先回答已知部分。不承诺必赔或必能投保。
简单题一两句，复合题每问一个短结论；争取约70字，这是表达目标，必要条件不能为了字数删掉，也不能截断。
如用户消息要求JSON槽位，遵守格式：conclusion只说该项直接结论，condition只说必要限定，共同条件仅放第一项；其他槽位condition可以为空。
句式示例（只学习取舍，事实仍以本轮资料为准）：
问：免赔额每年重算吗？答：是，一般医疗每年有1万元免赔额。不要再讲年度保额、医院和加油包。
问：投保后什么时候生效？答：投保成功后次日零时生效。不要主动介绍疾病等待期；生效与等待期是不同问题。
问：停售后还可以续保吗？答：续保规则我无法确认，可以转专员核实。不要添加价格和保障介绍。'''

def main():
    chunks = corpus(False)
    client = cloud.llm_client()
    pipe = Pipeline(HybridRetriever(chunks, load(chunks, 'clauses'), client=client), client=client, direct_answer=False)
    cases = [items()[i] for i in (1, 3, 6, 10, 18, 26)] + [
        {'id': 'S01', 'question': '药省保每月最多报多少'},
        {'id': 'S02', 'question': '特定药品要在哪里购买'},
    ]
    with ThreadPoolExecutor(2) as pool:
        prepared = list(pool.map(lambda row: prepare(pipe, client, row), cases))
    original_sys, original_voice = pipeline._ANSWER_SYS, pipeline.VOICE_RULES
    result = {'model': pipe.slow_model, 'n_cases': len(cases), 'rounds': 2,
              'prompts': {'current': original_sys + original_voice, 'minimum': MINIMUM_SYS}, 'rows': []}
    path = ROOT / 'experiments/results/exp24_ds_minimum_prompt.json'
    try:
        for round_ in (1, 2):
            for variant in (('current', 'minimum') if round_ == 1 else ('minimum', 'current')):
                pipeline._ANSWER_SYS = original_sys if variant == 'current' else MINIMUM_SYS
                pipeline.VOICE_RULES = original_voice if variant == 'current' else ''
                for case, u in zip(cases, prepared):
                    t0 = time.perf_counter()
                    try:
                        a = pipe.answer(case['question'], voice=True, understanding=copy.deepcopy(u))
                        row = {'id': case['id'], 'round': round_, 'variant': variant,
                               'question': case['question'], 'text': a.text, 'chars': len(a.text),
                               'elapsed_s': time.perf_counter() - t0, 'route': a.route,
                               'attempts': a.attempts, 'fallback': a.fallback_used,
                               'violations': a.violations, 'timings': a.timings}
                    except Exception as exc:
                        row = {'id': case['id'], 'round': round_, 'variant': variant,
                               'error_type': type(exc).__name__, 'status_code': getattr(exc, 'status_code', None)}
                    result['rows'].append(row)
                    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                    print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        pipeline._ANSWER_SYS, pipeline.VOICE_RULES = original_sys, original_voice
    for variant in ('current', 'minimum'):
        rows = [r for r in result['rows'] if r['variant'] == variant and 'text' in r]
        print(variant, len(rows), statistics.median(r['chars'] for r in rows),
              statistics.median(r['elapsed_s'] for r in rows), flush=True)

if __name__ == '__main__':
    main()
