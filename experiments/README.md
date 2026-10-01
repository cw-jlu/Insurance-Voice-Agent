# 实验脚本

实验记录与结论见 [docs/06-实验记录.md](../docs/06-实验记录.md)。

## 环境

```bash
pip install -r requirements.txt
```

## 运行

在仓库根目录执行（Windows 下建议设置 `PYTHONIOENCODING=utf-8`）：

```bash
python experiments/exp1_quote.py       # 实验 1：报价全量枚举
python experiments/exp3_guardrail.py   # 实验 3：护栏（开发集 + 保留集 1 + 保留集 2）
python experiments/exp_retrieval.py    # 实验 2（检索部分）
python experiments/exp4_tts.py         # 实验 4（部分）：需要联网，使用 edge-tts
python experiments/exp5_interrupt.py   # 实验 5（离线部分）：约 5 分钟
python experiments/exp6_slots.py       # 实验 6（部分）：条件抽取
```

结果写入 `experiments/results/*.json`。`exp3_guardrail_v1.json` 是 v1 护栏的结果，用于对比。

## 文件

| 文件 | 内容 |
|---|---|
| `exp3_heldout.py` | 护栏保留集 1：v1 写完后编写，发现问题后参与了 v2 调整 |
| `exp3_heldout2.py` | 护栏保留集 2：v2 调整前编写，调整期间未查看结果，只用于最终评估 |
| `evidence.py` | 实验 3 共用的证据构造（与检索切片格式一致） |
