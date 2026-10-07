"""实验 4 扩大：端到端语音时延，代码版本与第六轮重测（exp4_e2e_v4_fix）相同，只扩大用例。

第六轮重测只有 12 句 × 2 轮 = 24 轮。本轮在原 12 句之外，运行前新写 18 句（12 句单项问题、6 句复合问题），
共 30 句 × 2 轮 = 60 轮。流程、计时口径与 exp4_e2e_v4.py 完全相同。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import exp4_e2e_v4 as v4  # noqa: E402

NEW = [
    ("报价2", "我妈五十八岁，没有社保，给她买一年多少钱？", {"relation": "父母"}),
    ("报价3", "按月交的话每个月多少？", {"relation": "本人", "age": 35, "has_social": True}),
    ("病史2", "我前年做过甲状腺结节手术，现在还能买吗？", {"relation": "本人", "age": 45, "has_social": True}),
    ("病史3", "我有糖尿病，住院能不能报？", {"relation": "本人", "age": 52, "has_social": True}),
    ("比例", "没有社保的话报多少？", {"relation": "本人", "age": 30, "has_social": False}),
    ("医院", "私立医院看病能报吗？", {"relation": "本人", "age": 30, "has_social": True}),
    ("续保", "明年还能接着保吗？", {"relation": "本人", "age": 30, "has_social": True}),
    ("优惠", "一家人一起买有没有优惠？", {"relation": "本人", "age": 33, "has_social": True}),
    ("身份2", "你这是录音吧，你是AI吗？", {}),
    ("免责", "喝酒以后出了车祸能赔吗？", {"relation": "本人", "age": 30, "has_social": True}),
    ("年龄上限", "我爸七十二了还能买吗？", {"relation": "父母", "age": 72, "has_social": True}),
    ("特药", "特药是什么意思，怎么报？", {"relation": "本人", "age": 40, "has_social": True}),
    ("复合5", "我有社保，住院花了三万，免赔额怎么扣，最后能拿回多少？", {"relation": "本人", "age": 35, "has_social": True}),
    ("复合6", "孩子住院有没有津贴，门急诊能不能报？", {"relation": "子女", "age": 6, "has_social": True}),
    ("复合7", "确诊重疾以后，住院和外购药分别怎么报，有没有一次性给钱？", {"relation": "本人", "age": 45, "has_social": True}),
    ("复合8", "意外受伤住院要等待期吗，免赔额是多少？", {"relation": "本人", "age": 30, "has_social": True}),
    ("复合9", "质子重离子和CAR-T能报吗，额度是多少？", {"relation": "本人", "age": 50, "has_social": True}),
    ("复合10", "我老婆怀孕了，生孩子和产检能报吗，孩子出生后能买吗？", {"relation": "配偶", "age": 29, "has_social": True}),
]

_run_turn = v4.run_turn


def run_turn(name, text, known, pipe, user_tts, agent_tts, opener_pcm, rec_path):
    """记录本轮 Agent 侧 TTS 是否断线重连（重连耗时计入时延，单独标出以便剔除比较）。"""
    before = agent_tts.reconnects
    t = _run_turn(name, text, known, pipe, user_tts, agent_tts, opener_pcm, rec_path)
    t["tts_reconnects"] = agent_tts.reconnects - before
    return t


if __name__ == "__main__":
    v4.run_turn = run_turn
    v4.UTTERANCES = v4.UTTERANCES + NEW
    v4.TAG = sys.argv[1] if len(sys.argv) > 1 else "_expand"
    r = v4.main(rounds=2)
    for grp in ("all", "simple", "complex"):
        print(grp, r[grp])
    print("over 10s (all audio ready):", r["over_10s_all_ready"], "/", r["turns_ok"], "errors:", len(r["errors"]))
