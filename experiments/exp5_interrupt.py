"""实验 5（离线部分）：打断逻辑。

5a 语义层：区分附和与真打断。
5b 防回声：免提时 ASR 把 Agent 自己的播报识别回来，用"与当前播报的相似度"过滤。
5c 取消与播报账本：在流式生成 + 播放过程中随机时刻打断，检查账本是否准确、旧回答是否泄漏、
   纠正条件后是否按新条件重新报价。

声学层（VAD、真实回声消除）和真实 ASR 噪声需要接入音频链路后测，不在本实验范围。
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.dialogue import PlaybackLedger, SentState, classify_barge_in, echo_similarity, is_echo  # noqa: E402
from agent.quote import Insured, quote  # noqa: E402
from agent.speech import extract_slots  # noqa: E402

OUT = ROOT / "experiments" / "results"
random.seed(20261001)

# --- 5a ------------------------------------------------------------------------------------------
BACKCHANNEL = [
    "嗯", "嗯嗯", "哦", "对", "对对", "是", "是的", "好", "好的", "行", "可以", "明白", "知道了", "了解", "收到",
    "嗯，好的", "哦哦", "没错", "OK", "好吧", "这样啊", "嗯。", "对的", "啊", "噢",
]
INTERRUPT = [
    "等一下", "等等", "不对", "我没有社保", "不是，我是给我妈买", "我31岁", "那月缴多少", "你说慢点", "停一下",
    "我有高血压能买吗", "先别说了", "你是机器人吗", "那等待期多久", "我听不清", "说错了，我是35岁", "能便宜点吗",
    "我不需要", "我在开车", "那要怎么买", "稍等一下", "免赔额是多少", "打断一下", "我没听懂", "你听我说", "孩子能买吗",
]


def exp5a() -> dict:
    wrong = []
    for u in BACKCHANNEL:
        if classify_barge_in(u) != "backchannel":
            wrong.append({"utterance": u, "expected": "backchannel", "got": classify_barge_in(u)})
    for u in INTERRUPT:
        if classify_barge_in(u) != "interrupt":
            wrong.append({"utterance": u, "expected": "interrupt", "got": classify_barge_in(u)})
    false_stop = sum(1 for w in wrong if w["expected"] == "backchannel")
    missed = sum(1 for w in wrong if w["expected"] == "interrupt")
    return {"backchannel": len(BACKCHANNEL), "interrupt": len(INTERRUPT),
            "false_stop_rate": round(false_stop / len(BACKCHANNEL), 4),
            "interrupt_recall": round(1 - missed / len(INTERRUPT), 4), "errors": wrong}


# --- 5b ------------------------------------------------------------------------------------------
AGENT_SENTENCES = [
    "总保额是600万。", "最常用的是两块，一般住院300万、重大疾病医疗300万。", "要注意一般住院一年有1万免赔额。",
    "能不能买，要先在投保时做健康告知，以核保结果为准。", "高血压以及它引起的住院属于既往症，是不赔的。",
    "有社保的话，基础版一年318元，也可以按月交，每月31元。", "不是固定的，每年按投保时的年龄段计算。",
    "如果是重大疾病，没有1万免赔额。", "我把产品介绍和投保链接发到您手机上。", "投保成功后次日零时生效。",
]
_NOISE_CHARS = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处府研"


def asr_noise(text: str, drop: float, sub: float) -> str:
    out = []
    for ch in text:
        r = random.random()
        if r < drop:
            continue
        if r < drop + sub:
            out.append(random.choice(_NOISE_CHARS))
        else:
            out.append(ch)
    return "".join(out)


def echo_fragment(sentence: str, drop: float, sub: float) -> str:
    """Echo is heard mid-playback, so ASR usually returns a fragment of the sentence."""
    s = sentence.strip("。")
    n = len(s)
    ln = random.randint(min(4, n), n)
    st = random.randint(0, n - ln)
    return asr_noise(s[st : st + ln], drop, sub)


def exp5b() -> dict:
    noise_levels = {"clean": (0.0, 0.0), "light": (0.05, 0.10), "heavy": (0.10, 0.25)}
    echoes = {name: [(s, echo_fragment(s, *lv)) for s in AGENT_SENTENCES for _ in range(30)] for name, lv in noise_levels.items()}
    # Real user speech while the agent is talking: pair every interrupt with every agent sentence.
    users = [(s, u) for s in AGENT_SENTENCES for u in INTERRUPT]
    user_scores = [echo_similarity(u, s) for s, u in users]
    sweep = {}
    for th in (0.4, 0.5, 0.6, 0.7, 0.8):
        row = {f"echo_caught_{k}": round(sum(is_echo(e, s, th) for s, e in v) / len(v), 4) for k, v in echoes.items()}
        row["user_wrongly_dropped"] = round(sum(is_echo(u, s, th) for s, u in users) / len(users), 4)
        row["user_wrongly_dropped_without_correction_rule"] = round(sum(x >= th for x in user_scores) / len(user_scores), 4)
        sweep[th] = row
    worst_users = sorted(((echo_similarity(u, s), u, s) for s, u in users), reverse=True)[:5]
    return {"echo_samples_per_level": len(echoes["clean"]), "user_pairs": len(users), "threshold_sweep": sweep,
            "highest_scoring_real_user_utterances": [{"score": round(a, 3), "user": u, "playing": s} for a, u, s in worst_users]}


# --- 5c ------------------------------------------------------------------------------------------
async def one_trial(interrupt_after_s: float, gen_s: float, play_s_per_char: float) -> dict:
    """Agent is quoting 30岁有社保; user interrupts with '我没有社保' at a random moment."""
    slots = {"age": 30, "has_social": True}
    q = quote(Insured(slots["age"], slots["has_social"]), "年缴")
    m = quote(Insured(slots["age"], slots["has_social"]), "月缴")
    answer = [f"有社保的话，基础版一年{q.total:g}元。", f"也可以按月交，每月{m.total:g}元。", "一年下来年缴更划算。",
              "投保成功后次日零时生效。", "首年有30天等待期。"]

    ledger = PlaybackLedger()
    queue: asyncio.Queue = asyncio.Queue()
    turn = [1]
    stale_played: list[str] = []
    emitted_after_cancel = 0
    cancelled = asyncio.Event()

    async def generate(turn_id: int, sentences: list[str]):
        nonlocal emitted_after_cancel
        for s in sentences:
            await asyncio.sleep(gen_s)
            if turn[0] != turn_id:
                emitted_after_cancel += 1
            await queue.put((turn_id, s))

    async def play():
        while True:
            turn_id, text = await queue.get()
            if turn_id != turn[0]:
                stale_played.append(text)
                continue
            s = ledger.add(text)
            s.state = SentState.PLAYING
            await asyncio.sleep(len(text) * play_s_per_char)
            if s.state == SentState.PLAYING:
                s.state = SentState.PLAYED

    gen_task = asyncio.create_task(generate(1, answer))
    play_task = asyncio.create_task(play())
    await asyncio.sleep(interrupt_after_s)

    # --- barge-in ---
    t0 = time.perf_counter()
    gen_task.cancel()
    play_task.cancel()
    for t in (gen_task, play_task):
        try:
            await t
        except asyncio.CancelledError:
            pass
    cancel_ms = (time.perf_counter() - t0) * 1000
    heard, dropped = ledger.interrupt()
    while not queue.empty():  # sentences generated but not yet played are discarded with the old turn
        dropped.append(queue.get_nowait()[1])
    turn[0] = 2

    # --- update slot from the user's correction and re-quote ---
    new = extract_slots("我没有社保")
    slots.update(new)
    q2 = quote(Insured(slots["age"], slots["has_social"]), "年缴")
    m2 = quote(Insured(slots["age"], slots["has_social"]), "月缴")
    answer2 = [f"明白，没有社保的话一年{q2.total:g}元，按月是每月{m2.total:g}元。"]
    gen2 = asyncio.create_task(generate(2, answer2))
    play2 = asyncio.create_task(play())
    await gen2
    await asyncio.sleep(len(answer2[0]) * play_s_per_char + 0.01)
    play2.cancel()
    try:
        await play2
    except asyncio.CancelledError:
        pass

    played_turn2 = [s.text for s in ledger.sentences if s.text in answer2 and s.state == SentState.PLAYED]
    states = [s.state for s in ledger.sentences]
    return {
        "cancel_ms": cancel_ms,
        "heard": heard,
        "dropped": dropped,
        # Heard must be a prefix of the answer (in order), and nothing heard may also be dropped.
        "ledger_consistent": all(st in (SentState.PLAYED, SentState.DROPPED) for st in states)
        and heard == answer[: len(heard)] and not (set(heard) & set(dropped))
        and heard + dropped == answer[: len(heard) + len(dropped)],
        "stale_played": len(stale_played),
        "emitted_after_cancel": emitted_after_cancel,
        "new_quote_correct": played_turn2 == answer2 and "679" in answer2[0] and "67" in answer2[0],
        "old_number_after_interrupt": any("318" in s or "每月31" in s for s in played_turn2),
    }


async def exp5c(trials: int = 300) -> dict:
    results = []
    for _ in range(trials):
        # Generation (~1 sentence / 20 ms) is faster than playback (~10 ms per char, i.e. ~150 ms per
        # sentence), as in the real system. Interrupts land anywhere within ~0.9 s.
        results.append(await one_trial(interrupt_after_s=random.uniform(0.005, 0.9), gen_s=0.02, play_s_per_char=0.01))
    cancel = sorted(r["cancel_ms"] for r in results)
    return {
        "trials": trials,
        "ledger_consistent": sum(r["ledger_consistent"] for r in results),
        "stale_sentence_played": sum(r["stale_played"] for r in results),
        "emitted_after_cancel": sum(r["emitted_after_cancel"] for r in results),
        "new_quote_correct": sum(r["new_quote_correct"] for r in results),
        "old_number_spoken_after_interrupt": sum(r["old_number_after_interrupt"] for r in results),
        "interrupt_points": {
            "before_any_speech": sum(1 for r in results if not r["heard"]),
            "mid_answer": sum(1 for r in results if r["heard"] and len(r["heard"]) < 5),
            "after_whole_answer_heard": sum(1 for r in results if len(r["heard"]) == 5),
        },
        "cancel_ms_p50": round(cancel[len(cancel) // 2], 3),
        "cancel_ms_p99": round(cancel[int(len(cancel) * 0.99) - 1], 3),
        "note": "cancel_ms 只包含进程内取消耗时；真实停播时延还取决于 VAD 判定（约 300ms）和客户端音频缓冲。",
    }


def main() -> dict:
    result = {"5a_semantic": exp5a(), "5b_echo": exp5b(), "5c_cancel_ledger": asyncio.run(exp5c())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp5_interrupt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
