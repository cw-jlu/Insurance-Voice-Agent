"""实验 4（部分）：句级流式播报对首包时延的影响。

4a 离线：LLM 输出到达多少字时，才能交给 TTS 第一段（有/无 60 字软切分）。
4b 联网：用 edge-tts（免费的微软云端 TTS，作为阿里云 TTS 接入前的替身）实测
   "整段合成"与"只合成第一句"的首个音频包时延。云服务不同，绝对值会不同，但能说明句级流式的收益。
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.speech import SentenceFeeder  # noqa: E402

OUT = ROOT / "experiments" / "results"
VOICE = "zh-CN-XiaoxiaoNeural"

ANSWERS = {
    "a": "总保额是600万。最常用的是两块：一般住院300万、重大疾病医疗300万，还有癌症特药和质子重离子这类先进疗法。"
         "要注意一般住院一年有1万免赔额，要在二级及以上公立医院普通部看病。您更关心住院还是大病？",
    "b": "这个要分开说。能不能买，要先在投保时做健康告知，以核保结果为准，我这边没法替核保下结论。"
         "如果能买，高血压以及它引起的住院属于既往症，是不赔的；和高血压无关的疾病或意外住院，过了30天等待期，符合条件就能报。"
         "投保时一定要如实告知，不然会影响以后理赔。",
    "c": "您有社保吗？有社保的话，基础版一年318元，也可以按月交，每月31元，一年下来年缴更划算。",
    "d": "不是固定的，每年按投保时的年龄段计算。比如26到30岁一年318元，到31岁就是423元。",
    "e": "不完全是。一般住院是一年累计超过1万的部分按100%报，但如果按有社保投保、看病时却没用社保结算，只报60%。"
         "如果是重大疾病，没有1万免赔额。另外要在二级及以上公立医院普通部、是合理必要的费用、过了30天等待期、不是既往症；"
         "医保已经报了的部分不会重复报。",
    # 一个首句很长、迟迟没有句号的回答，用来体现软切分的作用
    "long_first": "关于您问的这几项保障，一般住院、门急诊、重大疾病医疗、特定疾病康复住院、恶性肿瘤先进疗法、特定药品，"
                  "还有异地转诊交通住宿和重疾护工，这些都在必选计划里，加油包是另外选的。",
}


def chars_until_first_piece(text: str, soft_flush: int | None) -> int:
    f = SentenceFeeder(soft_flush=soft_flush)
    for i, ch in enumerate(text, 1):
        if f.feed(ch):
            return i
    return len(text)


def exp4a() -> dict:
    rows = {}
    for k, t in ANSWERS.items():
        rows[k] = {
            "answer_chars": len(t),
            "first_piece_after_chars_soft60": chars_until_first_piece(t, 60),
            "first_piece_after_chars_no_soft": chars_until_first_piece(t, None),
        }
    return rows


async def ttfb(text: str) -> tuple[float, float]:
    """(time to first audio chunk, time to last chunk), seconds."""
    import edge_tts

    t0 = time.perf_counter()
    first = None
    async for chunk in edge_tts.Communicate(text, VOICE).stream():
        if chunk["type"] == "audio" and first is None:
            first = time.perf_counter() - t0
    return first or float("nan"), time.perf_counter() - t0


async def exp4b(rounds: int = 3) -> dict:
    out = {}
    for k, t in ANSWERS.items():
        first_sentence = SentenceFeeder(60).feed(t)[0]
        whole, first = [], []
        for _ in range(rounds):
            whole.append(await ttfb(t))
            first.append(await ttfb(first_sentence))
        out[k] = {
            "whole_text_first_audio_s": round(statistics.median(x[0] for x in whole), 3),
            "whole_text_done_s": round(statistics.median(x[1] for x in whole), 3),
            "first_sentence": first_sentence,
            "first_sentence_first_audio_s": round(statistics.median(x[0] for x in first), 3),
        }
    return out


def main() -> dict:
    result = {"4a_offline_first_piece": exp4a()}
    try:
        result["4b_edge_tts"] = asyncio.run(exp4b())
        result["4b_note"] = "edge-tts 为替身服务；正式方案用阿里云流式 TTS，需接入后复测。"
    except Exception as e:  # network may be unavailable
        result["4b_edge_tts_error"] = repr(e)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp4_tts.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
