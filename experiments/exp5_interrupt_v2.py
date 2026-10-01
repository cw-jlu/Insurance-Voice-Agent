"""实验 5 扩充：打断逻辑。

5a 语义层：约 240 句，附和与真打断由模板组合生成（固定种子），包含第一轮没有的说法。
5b 防回声：Agent 播报句改用实验 2 中流水线的真实回答（按句切分），用户插话用 5a 的真打断句。
5c 取消与播报账本：1000 次随机打断。
第一轮的规则不针对本轮用例修改，结果即首次运行结果。
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agent.dialogue import classify_barge_in, echo_similarity, is_echo  # noqa: E402
from agent.speech import SentenceFeeder  # noqa: E402
import exp5_interrupt as v1  # noqa: E402

OUT = ROOT / "experiments" / "results"
rng = random.Random(20261005)

BC_CORE = ["嗯", "嗯嗯", "嗯哼", "哦", "哦哦", "噢", "啊", "对", "对对对", "对的", "是", "是的", "是是", "好", "好的", "好好",
           "行", "行行", "可以", "明白", "明白了", "知道了", "了解", "收到", "没错", "OK", "好吧", "这样啊", "原来如此", "嗯对", "然后呢", "你说"]
INT_STOP = ["等一下", "等等", "停一下", "先停", "先别说", "打住", "稍等", "慢点说", "你先听我说", "我插一句", "不好意思打断一下"]
INT_FIX = ["不对，我没有社保", "不是，我是给我妈买", "说错了，我是35岁", "我不是30岁，是40岁", "我没社保", "不是我，是给我爸买",
           "错了，是按月交", "我有社保的", "我女儿不是8岁，是10岁", "不对不对，我有医保"]
INT_Q = ["那月缴多少", "免赔额是多少", "等待期多久", "孩子能买吗", "高血压能买吗", "你是机器人吗", "怎么买", "能便宜点吗",
         "门诊能报吗", "那重疾呢", "什么时候生效", "能给我爸妈买吗"]
INT_REJ = ["我不需要", "我在开车", "不用了谢谢", "没兴趣", "我在忙", "以后再说吧", "别打了"]


def build_5a() -> list[tuple[str, str]]:
    cases = []
    for _ in range(120):
        k = rng.randint(1, 2)
        cases.append(("，".join(rng.sample(BC_CORE, k)), "backchannel"))
    pool = INT_STOP + INT_FIX + INT_Q + INT_REJ
    for _ in range(120):
        u = rng.choice(pool)
        if rng.random() < 0.3:
            u = rng.choice(["嗯，", "哦，", "那个，", "喂，"]) + u
        cases.append((u, "interrupt"))
    return cases


def exp5a(cases):
    errs = [{"u": u, "expected": e, "got": classify_barge_in(u)} for u, e in cases if classify_barge_in(u) != e]
    bc = [c for c in cases if c[1] == "backchannel"]
    it = [c for c in cases if c[1] == "interrupt"]
    false_stop = sum(1 for e in errs if e["expected"] == "backchannel")
    missed = sum(1 for e in errs if e["expected"] == "interrupt")
    return {"backchannel": len(bc), "interrupt": len(it), "false_stop_rate": round(false_stop / len(bc), 4),
            "interrupt_recall": round(1 - missed / len(it), 4), "error_examples": errs[:20]}


def agent_sentences() -> list[str]:
    path = OUT / "exp2_qa_r3.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    out = []
    for r in rows:
        f = SentenceFeeder()
        out += [s for s in f.feed(r["pipeline"]["text"]) + f.flush() if len(s) >= 6]
    return out


def exp5b(sents, users):
    levels = {"clean": (0.0, 0.0), "light": (0.05, 0.10), "heavy": (0.10, 0.25)}
    echoes = {k: [(s, v1.echo_fragment(s, *lv)) for s in sents for _ in range(3)] for k, lv in levels.items()}
    pairs = [(rng.choice(sents), u) for u in users for _ in range(20)]
    sweep = {}
    for th in (0.5, 0.6, 0.7):
        row = {f"echo_caught_{k}": round(sum(is_echo(e, s, th) for s, e in v) / len(v), 4) for k, v in echoes.items()}
        row["user_wrongly_dropped"] = round(sum(is_echo(u, s, th) for s, u in pairs) / len(pairs), 4)
        sweep[th] = row
    worst = sorted(((echo_similarity(u, s), u, s) for s, u in pairs if is_echo(u, s, 0.6)), reverse=True)[:8]
    return {"agent_sentences": len(sents), "echo_samples_per_level": len(echoes["clean"]), "user_pairs": len(pairs),
            "threshold_sweep": sweep, "wrongly_dropped_examples@0.6": [{"user": u, "playing": s, "score": round(a, 3)} for a, u, s in worst]}


def main():
    cases = build_5a()
    sents = agent_sentences()
    users = sorted({u for u, e in cases if e == "interrupt"})
    res = {"5a_semantic": exp5a(cases), "5b_echo": exp5b(sents, users)}
    random.seed(20261006)
    res["5c_cancel_ledger"] = asyncio.run(v1.exp5c(trials=1000))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp5_interrupt_v2.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps(r, ensure_ascii=False, indent=1)[:6000])
