"""测试集质检（在运行任何结果之前完成）：标记无效题目，不删除，写明原因。"""

import json
import re
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "experiments" / "data" / "qa_v2.json"

# 改写后语义漂移、角色颠倒或残留占位符的题（逐条人工检查）
BAD_PARAPHRASE = {
    "门急诊加油包多大年纪能买": "改写成了坐席反问用户年龄",
    "多大年纪可以买": "改写成了坐席反问",
    "你是不是录音": "改写成了坐席的回答",
    "大病以后的康复治疗管不管": "残留括号备注",
    "我妈60岁一年多少": "加入多个反问，语义混乱",
    "怀孕检查报不报": "残留占位符 XX岁",
    "全家5个人一起买有优惠吗": "新增'有人住过院'，改变了标准答案",
    "中风后去康复医院住院能报吗": "改写成坐席复述",
    "多少钱一年": "改写成坐席复述",
    "看牙能报吗": "改写成坐席的回答",
    "癌症用药前的基因检测能报多少": "改写成坐席复述",
    "在药店自己买的感冒药能报吗": "主语变成'您'，角色不清",
}


def main():
    items = json.loads(PATH.read_text(encoding="utf-8"))
    by_id = {i["id"]: i for i in items}
    n_bad = 0
    for it in items:
        it["valid"], it["invalid_reason"] = True, ""
        q = it["question"]
        # 生成器缺陷：配偶年龄过小、本人未成年自述投保
        m = re.match(r"(我老婆|我老公)(\d+|[一二两三四五六七八九十]+)岁", q)
        if m and m.group(2).isdigit() and int(m.group(2)) < 18:
            it["valid"], it["invalid_reason"] = False, "生成器缺陷：配偶年龄不合理"
        m = re.match(r"我(\d+)岁", q)
        if m and int(m.group(1)) < 18 and it["category"] == "价格":
            it["valid"], it["invalid_reason"] = False, "生成器缺陷：未成年人自述投保"
        if it["source"] == "改写":
            orig = by_id[it["paraphrase_of"]]
            if orig["question"] in BAD_PARAPHRASE:
                it["valid"], it["invalid_reason"] = False, "改写无效：" + BAD_PARAPHRASE[orig["question"]]
    for it in items:  # a paraphrase of an invalid base item is invalid too
        if it["source"] == "改写" and not by_id[it["paraphrase_of"]]["valid"]:
            it["valid"], it["invalid_reason"] = False, "原题无效"
        n_bad += not it["valid"]
    PATH.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    import collections
    v = [i for i in items if i["valid"]]
    print("total", len(items), "invalid", n_bad, "valid", len(v), collections.Counter(i["split"] for i in v),
          collections.Counter(i["source"] for i in v))
    for i in items:
        if not i["valid"]:
            print(" ", i["id"], i["question"][:30], "|", i["invalid_reason"])


if __name__ == "__main__":
    main()
