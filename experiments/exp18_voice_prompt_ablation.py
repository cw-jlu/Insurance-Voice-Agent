"""Paired voice-prompt comparison with shared understanding and retrieval.

No answer is shortened after generation. This screens prompts for answer quality and
length; finalists still need a real TTS/playback run before any 10-second claim.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from agent import cloud, pipeline, small_model  # noqa: E402
from agent.pipeline import HybridRetriever, Pipeline, combine_gaps, gaps_to_check  # noqa: E402
from data_complex import items as complex_items  # noqa: E402
from embed_cache import corpus, load  # noqa: E402
from exp2_qa_v2 import model_judge, regex_judge  # noqa: E402

ORIGINAL_ANSWER_SYS = pipeline._ANSWER_SYS


COMPACT_SYSTEM = """你是众安保险的AI语音坐席，回答尊享e生·百万医疗保险2026版的问题。
只依据【资料】和【报价结果】作答；数字必须有同一责任的依据，不能猜测、跨责任套规则或承诺必赔。
先辨认用户问了几件事，每件都要回答。用电话口语给最短的完整结论，再讲直接影响结论的条件；不念未问的保额、流程、医院或产品目录。
免赔额、比例、等待期、责任归属和加油包是否另购必须说对。基础版与加油包分开说，不能把某一责任的规则当成全部责任通用。
说赔付比例时要区分：按有社保身份投保且就诊未用社保结算，适用资料里的降低比例；按无社保身份投保则用资料里的正常比例。投保身份不明而用户只说没用社保结算时，条件化说明，不得直接断言100%。
涉及病史须提示健康告知，能否投保以核保为准；既往症不能承诺赔付。理赔流程、清单等资料没有的事项，说明无法确认并可转专员，不编造。
ICU住院医疗费用与另购的ICU津贴不同；特药社保目录内的报销规则不能套用到一般医疗。
【必须原样包含的话】如有内容，应在不改变含义的前提下包含。不要Markdown、编号、铺垫和通用销售提醒。"""

COMPACT_SYSTEM_V2 = COMPACT_SYSTEM + """
额外优先级：资料能回答的部分必须先明确回答，不能因另一部分清单或流程无法确认，就只念转专员兜底；【必须原样包含的话】放在已知结论后。
只问基础版时，只用基础版责任回答；不要因检索结果出现加油包就主动改说加油包的免赔额、等待期或保额。用户点名加油包才讲它。
问具体能报多少钱时，先算费用乘比例，再受单次或年度限额；只应用该责任资料里的比例，不把另一责任的社保条件套进来。"""

HUMAN_EXAMPLES = """\n电话短答示例只用于学习句式；实际事实、数字和适用条件仍必须以本题【资料】为准，不能照搬示例：
问：以后每年都是这个价吗？答：不会固定，保费会随年龄变化。
问：自己先付一万，剩下全报吗？答：一般医疗先扣1万免赔，余额按约定比例赔；有社保身份投保却没用社保结算，比例会降低。
问：住院和门急诊怎么报，ICU有补贴吗？答：住院和门急诊共用免赔额；ICU津贴得另购。若用户问具体金额，再补资料里的数字。
问：一部分能确定、清单不全怎么办？答：先说已能确定的结论；清单以条款为准，可帮您转专员确认。"""


PROMPTS = {
    "current": pipeline.VOICE_RULES,
    "human": ("\n8. 像有经验的人工坐席打电话：先直接回答用户问到的每一件事，每件事一句短话；"
              "只带上决定结论的条件，不朗读证据、推导、保额目录或没有被问到的产品信息。"
              "问‘都保什么’就概括主要保障类别，不逐项介绍规则；问金额先说能确定的结果，再说影响结果的一个关键条件。"
              "答完即停，不加通用提醒或销售引导。内部检查：每个子问题是否都答到了，免责/加油包/核保条件是否保留。"),
    "checklist": ("\n8. 生成前在心里把用户的问题拆成待回答的小项；逐项用最短的完整口语句写答案，"
                  "每项只保留结论和不可省的适用条件。写完再在心里逐项核对，不漏任何一项、不把某项责任的规则套到全部责任。"
                  "概览问题只列主要类别；数额问题先给结论，不展开计算过程；资料不足就明确缺哪项条件。"
                  "不要念条款目录、重复数字、复述问题，也不要附加无关的销售话术。"),
    "editor": ("\n8. 电话快答编辑法：把‘在这种情况下’改成‘若’，把‘另外还需要注意的是’删掉，"
               "去掉同一数字的重复解释；但不能删任何被问到的事项、影响结论的条件或必要免责。"
               "先报结论，再报关键限制。多问题逐项各一句；概览只概括类别，细则留到用户追问。"
               "不逐行复述资料，不加与本题无关的提醒。最后自查是否覆盖全部子问题。"),
    "soft45": ("\n8. 这是电话快答。先在心里列出用户问到的每一项，再用尽量少的字逐项直接回答，"
               "通常争取约45个汉字；这是表达目标，不是截断上限，若必要条件或多问答案放不下，必须完整保留。"
               "只说结论、直接影响结论的条件与必要免责；不报用户没问的保额、流程、年龄、医院或其他附加保障。"
               "概览题概括主要类别，不逐项读条款。答完即停。"),
    "essential": ("\n8. 请像忙碌但负责的人工坐席快答：每句话都必须直接回答一个子问题，或交代一个会改变答案的关键条件；"
                 "若一句话不属于这两类，就不要说。每个子问题都要答，不能用笼统概括替代。"
                 "不复述用户提问，不列全部保障细则，不解释推导，不主动介绍其他产品。"
                 "保留适用责任、加油包需另购、核保或免责等会改变结论的信息。"),
    "compact_plan": ("\n8. 语音回复只说‘结论＋关键条件’，不要写说明文。先默想每个子问题的最短准确答案，"
                     "再把相同条件合并说一次；用户问几项就答几项，不得漏答。"
                     "优先用短语并列：‘A能报，B不能；C要另购。’这种结构仅是句式示意，不是产品事实。"
                     "概览只说类别，数字只说与提问直接相关的，不重复保额与比例。"),
    "telegraph": ("\n8. 用自然的电话短句回答，不念资料名称和责任全称，直接说‘住院’‘康复’‘药品’等口语简称。"
                  "多个问题用分号并列，每个问题只说结论、被问的数字和一个会改变结论的必要条件；共同条件合并说一次。"
                  "不复述用户的年龄、花费或病情，不解释推导，不介绍未问的保额和细则，不加销售引导。"
                  "不能省略任何子问题、另购要求、免责或核保条件。"),
    "minimum": ("\n8. 只输出让用户得到准确答案所需的最小信息：每个子问题都有明确结论，"
                "只有会改变这个结论的条件才补充；其他正确但未被问到的事实一律不说。"
                "把同一责任的数字合并、共享限制只说一次，不念责任全名、资料出处、用户已说的背景，"
                "不用‘需要注意的是’‘具体来说’等铺垫。不得删去免责、另购、核保和多问答案。"),
    "soft70": ("\n8. 用电话里一口气能说完的短答。复合问题争取约70个汉字、简单问题更短；"
               "这是写作目标而非截断上限，用户问到的每一项及必要条件都必须保留。"
               "使用并列短句合并共同条件，不逐项念责任全名、保额、医院、上下限或计算过程，除非用户正问这些。"
               "只说结论和关键限定，额外细则等追问。"),
    "safe60": ("\n8. 电话快答，争取复合问题约60字、简单问题更短；这是生成时的软目标，不能删去用户问到的事项或必要条件。"
               "每个子问题给结论，共用的限制只说一次，责任用口语简称，不念未问的保额、流程与条款。"
               "报销比例必须与具体责任、投保社保身份、就诊结算方式配对；只说‘没走社保’而投保身份不明时，"
               "须说明‘按有社保身份投保却未用社保结算’才适用降低比例，不能直接说100%。"
               "年龄上下限按数值比较，不能把范围内的人说成超龄。答完即停。"),
    "safe80": ("\n8. 像人工客服做准确快答：先用最短口语回答每个子问题，再补一两个直接改变结论的条件。"
               "不复述用户背景，不念责任全称、没问的保额或推导；共同条件合并讲，额外细节等追问。"
               "说赔付比例时分别核对投保时有无社保、就诊时是否用社保结算；若投保身份未给，必须条件化说清，"
               "不得混淆60%与100%。年龄区间按数值核对。尽量一口气说完，但不可遗漏任何子问和必要免责。"),
    "base_min": "\n语音表达：先说每项结论，再说直接决定结论的条件；不用责任全称和无关数字。",
    "base_min45": ("\n语音表达：尽量用约45个汉字完成快答，这是软目标，不得遗漏任何子问或必要条件。"
                   "共享条件只说一次，答案用并列短句，不重复用户背景。"),
    "base_min70": ("\n语音表达：复合题尽量约70字、简单题更短；这是软目标，不截句。"
                   "每问一短句，保留必要数字与条件，不解释推导或额外介绍。"),
    "base_min70_v2": ("\n语音表达：复合题尽量约70字、简单题更短；这是软目标，不截句。"
                      "每问一短句，保留必要数字与条件，不解释推导或额外介绍。"),
    "fewshot_full": ("\n语音表达：沿用示例的短句风格，先把用户问的每件事答到，再说直接影响结论的条件。"
                     "不截断答案，不念没问的细则。" + HUMAN_EXAMPLES),
    "fewshot_compact": ("\n语音表达：沿用示例的短句风格，先把用户问的每件事答到，再说直接影响结论的条件。"
                        "不截断答案，不念没问的细则。" + HUMAN_EXAMPLES),
}

BASE_SYSTEMS = {name: COMPACT_SYSTEM for name in ("base_min", "base_min45", "base_min70")}
BASE_SYSTEMS["base_min70_v2"] = COMPACT_SYSTEM_V2
BASE_SYSTEMS["fewshot_compact"] = COMPACT_SYSTEM


def stats(values):
    if not values:
        return {"n": 0}
    xs = sorted(values)
    return {"n": len(xs), "p50": round(statistics.median(xs), 3),
            "p90": round(xs[int(0.9 * (len(xs) - 1))], 3), "max": round(xs[-1], 3)}


def rows_for(scope):
    rows = [{**row, "source": "complex"} for row in complex_items()] if scope not in ("v5", "v6") else []
    if scope in ("all", "v5", "v6"):
        filename = f"qa_{scope}.json" if scope in ("v5", "v6") else "qa_v7.json"
        source = f"qa_{scope}" if scope in ("v5", "v6") else "qa_v7"
        rows += [{**row, "source": source} for row in json.loads(
            (ROOT / "experiments" / "data" / filename).read_text(encoding="utf-8"))
            if row.get("valid") is not False and not row.get("turns")]
    return rows


def prepare(pipe, client, row):
    question = row["question"]
    known = dict(row.get("context") or {})
    for attempt in range(3):
        try:
            with ThreadPoolExecutor(max_workers=3) as pool:
                fu = pool.submit(pipe.understand, question, known)
                fg = pool.submit(small_model.detect_gaps, question, client)
                fr = pool.submit(pipe.retrieve, question)
                u, detected, retrieved = fu.result(), fg.result(), fr.result()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    u["_retrieved"] = retrieved
    u["_gaps_check"] = gaps_to_check(u.get("gaps", []), detected)
    u["gaps"], u["_gaps_judged"] = combine_gaps(u.get("gaps", []), detected, question), True
    if u.get("intent") == "缺口" and not u["gaps"]:
        u["intent"] = "保障" if u.get("items") else "其他"
    return u


def run_variant(pipe, client, prepared, name, workers):
    pipeline._ANSWER_SYS = BASE_SYSTEMS.get(name, ORIGINAL_ANSWER_SYS)
    prompt = PROMPTS[name]
    pipeline.VOICE_RULES = prompt

    def one(pair):
        row, u = pair
        known = dict(row.get("context") or {})
        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                answer = pipe.answer(row["question"], known, voice=True, agent_steps=0,
                                     understanding=copy.deepcopy(u))
                elapsed = time.perf_counter() - t0
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
        regex_ok, missing, forbidden = regex_judge(row, answer.text)
        try:
            judge_ok, reason = model_judge(row, answer.text, client)
        except Exception as exc:
            judge_ok, reason = None, type(exc).__name__
        return {"id": row["id"], "source": row["source"], "question": row["question"],
                "text": answer.text, "route": answer.route, "chars": len(answer.text),
                "answer_s": round(elapsed, 3), "regex_ok": regex_ok,
                "regex_missing": missing, "regex_forbidden": forbidden,
                "judge_ok": judge_ok, "judge_reason": reason}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(one, prepared))
    out.sort(key=lambda row: row["id"])
    judged = [row for row in out if row["judge_ok"] is not None]
    summary = {"n": len(out), "regex_ok": sum(row["regex_ok"] for row in out),
               "judge_ok": sum(row["judge_ok"] for row in judged),
               "both_ok": sum(row["regex_ok"] and row["judge_ok"] for row in judged),
               "judged_n": len(judged), "chars": stats([row["chars"] for row in out]),
               "answer_s": stats([row["answer_s"] for row in out])}
    return {"summary": summary, "rows": out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--scope", choices=("complex", "all", "v5", "v6"), default="complex")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--variants", nargs="+", choices=PROMPTS, default=list(PROMPTS))
    parser.add_argument("--tag", default="r1")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    client = cloud.llm_client()
    chunks = corpus(False)
    pipe = Pipeline(HybridRetriever(chunks, load(chunks, "clauses"), client=client), client=client)
    rows = rows_for(args.scope)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        prepared = list(pool.map(lambda row: (row, prepare(pipe, client, row)), rows))
    result = {"scope": args.scope, "n": len(rows), "variants": {}}
    try:
        for name in args.variants:
            result["variants"][name] = run_variant(pipe, client, prepared, name, args.workers)
            print(name, json.dumps(result["variants"][name]["summary"], ensure_ascii=False), flush=True)
            output = ROOT / "experiments" / "results" / f"exp18_voice_prompts_{args.tag}.json"
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        pipeline.VOICE_RULES = PROMPTS["current"]
        pipeline._ANSWER_SYS = ORIGINAL_ANSWER_SYS


if __name__ == "__main__":
    main()
