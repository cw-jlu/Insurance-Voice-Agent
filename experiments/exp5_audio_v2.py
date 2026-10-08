"""实验 5 扩大（真实音频打断）：第四轮每种条件只有 8 句打断 / 6 句附和 / 4 次只有回声，样本太小。

本轮在运行前新写用例（未用于任何调整）：
- 40 句真打断（纠正条件、叫停、提问、拒绝、身份），30 句附和，8 句 Agent 播报；
- 用户语音用两种音色（Ethan 男声、Serena 女声）交替；
- 三种回声条件：有回声消除（泄漏 3%）、回声消除不完全（10%）、无回声消除（30%）；
- 只有回声：8 句播报 × 每句 3 次（插话时刻随机），每种条件 24 次。
混音、ASR、判断流程与 exp5_audio.py 完全相同。
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agent import cloud  # noqa: E402
import exp5_audio  # noqa: E402

OUT = ROOT / "experiments" / "results"
exp5_audio.rng = random.Random(20261004)
rng = exp5_audio.rng

AGENT = [
    "您30岁有社保的话，基础版一年318元，按月交每月31元，一年下来年缴更划算一些。",
    "一般住院一年累计超过1万的部分，符合条件的按100%报，但要在二级及以上公立医院普通部看病。",
    "高血压属于既往症，相关的住院费用是不赔的，能不能投保要先做健康告知，以核保结果为准。",
    "重大疾病医疗没有1万免赔额，按100%报，一年最多300万，还包括住院前后的门急诊费用。",
    "等待期是30天，意外导致的不受等待期限制，续保的话也不再重新计算等待期。",
    "门急诊加油包是另外选购的，买了以后看门诊、急诊的费用也能按约定比例报销。",
    "如果您没有社保，或者有社保但没用社保结算，赔付比例是60%，所以看病时记得用社保卡。",
    "给孩子买的话，出生满28天到17岁都可以投保，价格比成年人便宜不少。",
]
INTERRUPTS = [
    "等一下，我没有社保", "不对不对，我四十二了", "是给我爸买，不是我", "我说的是我老婆",
    "停一下，你说慢点", "先别说了", "等会儿，我插一句", "你先听我说",
    "那按月交多少钱", "门诊能报吗", "什么叫免赔额啊", "那癌症呢，癌症能报吗",
    "我有糖尿病能买吗", "能便宜点吗", "孩子能一起买吗", "七十岁还能买吗",
    "我不需要，谢谢", "不用了，我已经有了", "我没兴趣", "别给我打电话了",
    "你是机器人吧", "你是真人吗", "转人工", "我要投诉你们",
    "等等，刚才那个价格是多少", "你刚说多少比例", "意外也要等三十天吗", "那我妈六十五能买吗",
    "我在开车，晚点再说", "我现在不方便", "你们是哪个公司的", "这个怎么理赔",
    "我上个月刚做过手术", "我有医保还要买这个吗", "这个和惠民保有什么区别", "能不能发个链接给我",
    "不是，我问的是住院", "哎你先停", "我听不太清", "那牙齿能报吗",
]
BACKCHANNELS = [
    "嗯", "嗯嗯", "对", "对对", "好", "好的", "行", "哦", "哦哦", "是", "是的", "明白",
    "了解", "知道了", "嗯，你说", "嗯嗯，然后呢", "好的好的", "对，没错", "嗯，我在听", "哦这样啊",
    "原来如此", "可以", "好，你继续", "嗯哼", "懂了", "对的", "行行", "啊，是吗", "嗯，好", "收到",
]
GAINS = ((0.03, "有回声消除"), (0.10, "回声消除不完全"), (0.30, "无回声消除"))
ECHO_REPEATS = 3


async def main_async():
    client = cloud.llm_client()
    agent_tts = cloud.RealtimeTTS(voice="Cherry")
    agent_audio = {s: cloud.resample_24k_to_16k(agent_tts.synth(s)[2]) for s in AGENT}
    agent_tts.close()
    user_audio = {}
    for i, voice in enumerate(("Ethan", "Serena")):
        tts = cloud.RealtimeTTS(voice=voice)
        for j, u in enumerate(INTERRUPTS + BACKCHANNELS):
            if j % 2 == i:
                user_audio[u] = (voice, cloud.resample_24k_to_16k(tts.synth(u)[2]))
        tts.close()
    jobs = []
    for gain, cond in GAINS:
        for u in INTERRUPTS:
            jobs.append((cond, "打断", u, rng.choice(AGENT), gain))
        for u in BACKCHANNELS:
            jobs.append((cond, "附和", u, rng.choice(AGENT), gain))
        for a in AGENT:
            for _ in range(ECHO_REPEATS):
                jobs.append((cond, "只有回声", None, a, gain))
    sem = asyncio.Semaphore(4)

    async def run(job):
        cond, kind, u, a, gain = job
        voice, upcm = user_audio[u] if u else (None, None)
        async with sem:
            for attempt in range(3):
                try:
                    r = await exp5_audio.trial(kind, u, agent_audio[a], a, upcm, gain, client)
                    r["condition"], r["voice"] = cond, voice
                    return r
                except Exception as e:
                    if attempt == 2:
                        return {"condition": cond, "kind": kind, "user": u, "error": repr(e)[:200]}
                    await asyncio.sleep(2)

    return await asyncio.gather(*(run(j) for j in jobs))


def summarize(rows):
    summary = {}
    for _, cond in GAINS:
        rs = [r for r in rows if r.get("condition") == cond and "error" not in r]
        it = [r for r in rs if r["kind"] == "打断"]
        bc = [r for r in rs if r["kind"] == "附和"]
        eo = [r for r in rs if r["kind"] == "只有回声"]
        lat = sorted(r["stop_latency_s"] for r in it if r["stop_latency_s"] is not None)
        summary[cond] = {
            "interrupt_stopped": f"{sum(r['stopped'] for r in it)}/{len(it)}",
            "stop_latency_s": {"p50": round(statistics.median(lat), 3),
                               "p90": round(lat[int(round(0.9 * (len(lat) - 1)))], 3),
                               "max": round(max(lat), 3), "n": len(lat)} if lat else None,
            "backchannel_wrongly_stopped": f"{sum(r['stopped'] for r in bc)}/{len(bc)}",
            "echo_only_wrongly_stopped": f"{sum(r['stopped'] for r in eo)}/{len(eo)}",
        }
    return summary


def main():
    rows = asyncio.run(main_async())
    res = {"summary": summarize(rows), "errors": [r for r in rows if "error" in r], "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp5_audio_v2.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps(r["summary"], ensure_ascii=False, indent=1))
    print("errors", len(r["errors"]))
    for row in r["rows"]:
        if "error" in row:
            continue
        bad = (row["kind"] == "打断" and not row["stopped"]) or (row["kind"] != "打断" and row["stopped"])
        if bad:
            print("  ✗", row["condition"], row["kind"], row["user"], row["decisions"][:3])
