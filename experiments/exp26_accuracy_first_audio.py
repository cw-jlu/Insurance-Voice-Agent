"""Strict quality comparison and real cloud ASR/TTS first-frame comparison."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments')]
from agent import cloud
from agent.pipeline import Pipeline, HybridRetriever
from embed_cache import corpus, load
from exp18_voice_prompt_ablation import rows_for
from exp2_qa_v2 import source_text, regex_judge

JUDGE = '''你是保险答案审核员，根据原问题、原始保险资料、参考要点审核实际答案。只输出JSON：
{"correct":true/false,"missing_answers":[],"missing_conditions":[],"incorrect_claims":[],"reason":"简短原因"}。
核心结论或被问事项遗漏、被问金额/额度遗漏、数字错误、跨责任套规则、编造、保证承保或赔付均不合格。
必须保留会改变当前结论的责任专属条件：加油包另购、护工指定/认可机构服务、特药指定药店与处方审核、家庭共享不含一般门急诊、降低比例的有社保投保身份前提。
不要要求罗列没有问的额度或所有通用医院/等待期细节，留白本身不是错误；但问具体医院/等待期时必须回答。
参考要点可能只列核心，原文优先。原文不支持的新增说法也算错。假设得病的保障问题不等于用户已有病史；明确病史不能承诺承保。
逐项检查所有用户问到的事项，不能仅因为包含部分关键词或说了以条款为准就判通过。'''

def save(path, result):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)

def grade(case, answer, judge_client):
    facts = case.get('facts', '')
    raw = cloud.chat([{'role':'system','content':JUDGE}, {'role':'user','content':
        f"用户问题：{case['question']}\n参考要点：{facts}\n【原文】\n{source_text(facts)}\n【实际回答】\n{answer}"}],
        model=cloud.QWEN_ANSWER_MODEL, client=judge_client, temperature=0, response_format={'type':'json_object'})
    return json.loads(raw)

def accuracy(pipes):
    path = ROOT / 'experiments/results/exp26_accuracy.json'
    result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'model': cloud.SLOW_MODEL, 'judge_model': cloud.QWEN_ANSWER_MODEL, 'judge_prompt':JUDGE,
        'scope':'30 complex + valid single-turn qa_v7', 'rows': []}
    judge_client = cloud.qwen_client()
    cases = rows_for('all')
    done = {(r['source'],r['id'],r['variant']) for r in result['rows'] if 'error_type' not in r}
    jobs = [(case,variant) for case in cases for variant in ('legacy','direct')
            if (case['source'],case['id'],variant) not in done]
    def one(job):
        case, variant = job
        row = {'id':case['id'], 'source':case['source'], 'variant':variant, 'question':case['question'], 'facts':case.get('facts')}
        try:
            a = pipes[variant].answer(case['question'], known=dict(case.get('context') or {}), voice=True)
            row.update(text=a.text, route=a.route, fallback=a.fallback_used, attempts=a.attempts,
                       violations=a.violations, timings=a.timings, chars=len(a.text))
            row['strict_judge'] = grade(case, a.text, judge_client)
            row['regex_ok'], row['regex_missing'], row['regex_forbidden'] = regex_judge(case, a.text)
        except Exception as exc:
            row['error_type'] = type(exc).__name__
        return row
    with ThreadPoolExecutor(3) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for f in as_completed(futures):
            row = f.result()
            result['rows'] = [r for r in result['rows'] if (r['source'],r['id'],r['variant']) != (row['source'],row['id'],row['variant'])]
            result['rows'].append(row)
            save(path, result)
            print(json.dumps({k:row.get(k) for k in ('id','source','variant','strict_judge','error_type')},ensure_ascii=False),flush=True)

def voice(pipes):
    from exp20_voice_overlap import trial, CASES
    from exp4_e2e import speech_end_s
    path = ROOT / 'experiments/results/exp26_first_audio.json'
    result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'model':cloud.SLOW_MODEL, 'tts_model':cloud.TTS_MODEL, 'voice':'Cherry', 'speech_rate':1.2,
        'vad_ms':700, 'measurement':'actual first PCM received; playback hardware not measured', 'rows':[]}
    # Existing cached identical synthetic user audio; same ASR settings in both arms, no acknowledgement filler.
    selected = [c for c in CASES if c[0] in ('等待期','复合','算术','概览')]
    tts = cloud.RealtimeTTS(voice='Cherry',speech_rate=1.2)
    try:
        tts.synth('您好。')  # identical warm persistent connection policy for both chains
        for round_ in (1,2):
            for variant in (('legacy','direct') if round_==1 else ('direct','legacy')):
                for name, question in selected:
                    key = (round_,variant,name)
                    if any((r['round'],r['variant'],r['case'])==key and 'error_type' not in r for r in result['rows']):
                        continue
                    row = {'round':round_,'variant':variant,'case':name,'question':question}
                    try:
                        audio_path = ROOT / 'experiments/recordings/exp20_inputs' / (hashlib.sha256(question.encode()).hexdigest()[:16]+'.pcm')
                        pcm = audio_path.read_bytes()
                        measured = trial(pipes[variant],pcm,tts,b'', 'stream',700,False)
                        row.update(measured)
                        row['first_pcm_available_s'] = measured['sentences'][0]['first_s'] - speech_end_s(pcm) if measured['sentences'] else None
                        row['audio_duration_s'] = sum(s['duration_s'] for s in measured['sentences'])
                        standards = {
                            '等待期': 'R01 意外住院不受疾病等待期限制；疾病住院首年等待期30天。',
                            '复合': 'R01 R02 R15 一般住院和一般门急诊属于基础版，共享年1万免赔和300万额度；正常比例100%，有社保身份投保却未用社保结算按60%；ICU津贴需另购，500元/天。',
                            '算术': 'R01 R02 急诊5000元与住院20000元符合责任时合计25000元，共享年度1万免赔；假设免赔未使用、无其他补偿且社保结算，余额15000元按100%赔；不能套用另购住院补偿加油包。',
                            '概览': 'R01 R02 R03 R04 R05 R06 R07 R08 基础版含一般住院/门急诊、重疾医疗、特定疾病康复住院、先进疗法、特药、重疾异地转诊交通住宿和护工；附加加油包需另购。概览不用逐项列全部细则。'}
                        row['strict_judge'] = grade({'question':measured['asr_text'], 'facts':standards[name]}, measured['text'], cloud.qwen_client())
                    except Exception as exc:
                        row['error_type'] = type(exc).__name__
                    result['rows'] = [r for r in result['rows'] if (r['round'],r['variant'],r['case'])!=key]
                    result['rows'].append(row)
                    save(path,result)
                    print(json.dumps({k:row.get(k) for k in ('case','round','variant','first_pcm_available_s','text','error_type')},ensure_ascii=False),flush=True)
    finally:
        tts.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('accuracy','voice'),required=True)
    args=ap.parse_args()
    client=cloud.llm_client()
    chunks=corpus(False)
    retriever=HybridRetriever(chunks,load(chunks,'clauses'),client=client)
    pipes={v:Pipeline(retriever,client=client,direct_answer=v=='direct')for v in ('legacy','direct')}
    try:
        (accuracy if args.mode=='accuracy' else voice)(pipes)
    except Exception as exc:
        print(json.dumps({'mode':args.mode,'startup_or_run_error_type':type(exc).__name__}),flush=True)
        raise SystemExit(1) from None

if __name__=='__main__':
    main()
