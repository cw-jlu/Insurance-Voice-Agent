"""给扩充测试集追加"换说法"题：用与生成模型不同的模型（qwen-turbo）把部分问题改写成新的口语说法，
标准答案不变，全部放入保留集，用于检验对未见过说法的泛化。只运行一次，结果写入 qa_v2.json。"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent import cloud  # noqa: E402

PATH = ROOT / "experiments" / "data" / "qa_v2.json"
MODEL = "qwen-turbo"  # 账号未开通其他厂商模型；与生成用的 qwen-plus 不同
SYS = ("把用户在电话里问保险的一句话，改写成另一种口语说法，意思和所有条件（年龄、有无社保、缴费方式、人数、病名）必须完全不变。"
       "可以换词、加语气词、调整语序，像真人随口说的。只输出改写后的一句话。")


def main():
    items = json.loads(PATH.read_text(encoding="utf-8"))
    if any(i["source"] == "改写" for i in items):
        print("already paraphrased"); return
    rng = random.Random(20261003)
    pool = [i for i in items if i["category"] != "多轮"]
    picked = rng.sample(pool, 45)
    client = cloud.llm_client()
    new = []
    for it in picked:
        q = cloud.chat([{"role": "system", "content": SYS}, {"role": "user", "content": it["question"]}],
                       model=MODEL, client=client, temperature=0.9).strip().strip("“”\"")
        n = dict(it)
        n.update({"id": f"P{len(new) + 1:03d}", "question": q, "source": "改写", "split": "heldout", "paraphrase_of": it["id"]})
        new.append(n)
        print(it["question"], "→", q)
    PATH.write_text(json.dumps(items + new, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(items) + len(new))


if __name__ == "__main__":
    main()
