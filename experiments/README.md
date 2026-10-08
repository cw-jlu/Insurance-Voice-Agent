# 实验脚本

## 当前运行模型

正式主回答默认使用DeepSeek官方接口`deepseek-flash`，关闭思考。
`src/agent/cloud.py`读取`.env`及Git忽略的`.env.deepseek`，需要`DEEPSEEK_API_KEY`，
接口默认`https://api.deepseek.com`，可用`DEEPSEEK_BASE_URL`配置。
理解/分类仍为`qwen-flash`，向量和TTS仍用百炼，原DASHSCOPE配置继续需要。
可用`ANSWER_PROVIDER`和`ANSWER_MODEL`明确设置供应商与主模型。
DeepSeek模式忽略旧`SLOW_MODEL=qwen-plus`；Qwen模式兼容原`SLOW_MODEL`设置。
恢复Qwen主回答可设`ANSWER_PROVIDER=qwen`、`ANSWER_MODEL=qwen-plus`，重启后生效。
exp20/exp21模型对照脚本明确选择供应商，Qwen对照与判分仍使用Qwen Plus。

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

## 提前准备、流式播报与句尾实验（exp20）

安装语音实验依赖：`pip install -r experiments/requirements-voice.txt`。
云端实测需要 `.env` 或环境变量中的 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、
`DASHSCOPE_WS_URL`、`TENCENT_APP_ID`、`TENCENT_SECRET_ID`、`TENCENT_SECRET_KEY`。
脚本缺配置时只输出变量名，不打印密钥。

```bash
python experiments/exp20_voice_overlap.py --limit 2 --vad-ms 500
python experiments/exp20_voice_overlap.py --vad-ms 300 500 700 --semantic-probe
python experiments/exp20_voice_overlap.py --vad-ms 700 --output exp20_qwen_final_700.json
python experiments/exp20_voice_overlap.py --vad-ms 700 --answer-provider deepseek --output exp20_deepseek_final_700.json
```

同一问题的用户音频只合成一次，分别实测整段回答、逐句流式、提前准备、提前准备加流式。
输入音频缓存在 `experiments/recordings/exp20_inputs/`，不同模型和后续复测复用相同音频。
DeepSeek 对照仅替换主回答模型，理解、分类、检索向量和 TTS 仍用原配置。
默认使用 `deepseek-flash`，关闭思考；可通过 `--answer-model` 指定其他模型。
密钥来自环境变量或 Git 忽略的 `.env.deepseek`，也支持在原 `.env` 中配置。
可用 `--cases` 选择问题，`--rounds` 重复测量；结果文件名用 `--output` 指定。
各组共用当前 `VOICE_RULES`：先给结论、逐项回答、保留必要条件、细则留待追问，无硬字数截断。
每组独立调用模型，生成仍有随机波动；结果保留全文、ASR 中间结果和逐句时间点，须逐题核查
改口后的社保条件、数字计算、附加险归属及复合问题完整性，不能只比较快慢。

提前准备最多启动三次，间隔至少 0.6 秒，在独立会话副本中执行；改口令旧候选失效。
初始逐次 partial 的策略可用 `--prepare-policy eager` 重现。当前默认 `pause` 策略在本地
检测到至少 120ms 停顿后准备，最小启动间隔为 0.2 秒，以减少长句过早消耗重算次数。
复用比较仅忽略标点/空白并规范化带单位的中文整数，保留小数点与否定词。
最终识别文本一致才复用并释放句子，否则重新处理。确认前不播业务结论，也不提交画像、
同意或链接状态。取消为协作式：分类调用可能继续完成，但完成后不再生成；流式生成在下个句子
边界停止并关闭连接。最终重算有独立工作线程，避免排在旧候选后面。

句尾实验比较 300/500/700ms 静音设置。语义判断只在本地检测到 200ms 停顿后进行影子调用，
记录判断耗时和是否先于 ASR 最终结果完成，不改变句尾或提前播报。影子判断应人工标记
是否误把半句当完整句，再决定是否接入。ASR 把一个问题切成多个最终句时，该轮记录为错误，
不计入正常时延；需另做连续语音和抢话测试。

结果文件 `experiments/results/exp20_voice_overlap.json` 会在每轮后保存。
首响包含预合成承接语；实质首响单独记录。播完时间按音频长度模拟，尚非客户端实播终点；
模板回答可能不调用生成模型，不能用它们单独证明流式生成的收益。

实现复查发现 `CallSession` 原先未把 `on_sentence` 传给 `Pipeline.answer`，现已修正。
`exp20_qwen_500.json` 为接线修正前的诊断结果，不应当作真正流式对照；
`exp20_qwen_fixed_500.json` 为中间版本，仍使用长句逗号切块。
当前流式路径只释放完整句，保留附加险另购提醒；护栏失败仍可能导致后续子问题未答，
质量必须逐题检查，不能据有首句交付就认定整个回答合格。

## 文件

## 全量模型对照（exp21）

`exp21_full_comparison.py --provider qwen --mode qa` 跑 qa_v2～qa_v7 的全部有效题及
30 道复合题（543 题，包含多轮上下文）；把 provider 改为 deepseek 跑主回答模型对照。
`--mode calls --rounds 3` 跑全部 36 种电话角色各三轮（108 通）。
理解、分类、检索和 TTS 均沿用 Qwen；DeepSeek 主回答使用 deepseek-flash、关闭思考。
自动判分和电话模拟用户统一用 Qwen Plus，存在评判偏差；判分通过不等于人工事实准确。
`--resume` 保留成功结果，只补未完成及执行失败项。每项执行结束保存，异常仅存类型。

语音完整四方案三轮：`exp20_voice_overlap.py --vad-ms 700 --rounds 3 --semantic-probe
--output exp21_qwen_voice.json`；DeepSeek 增加 `--answer-provider deepseek` 并换结果名。
语音仍是六种代表场景（72 轮/模型），不代表 543 道题全部走过音频链路。
打断回声使用 exp5_audio_v2 的全量 282 项混音场景；并非真实扬声器/麦克风回声消除验收。
`summarize_full_comparison.py` 汇总逐题自动评分、电话检查及模拟播完时间。
全量题库包含旧开发题和保留题，不应将合并成绩解释为全新保留集准确率。

电话报价评分已修复：识别 `回答:报价` 的更正报价，跳过没有金额的超龄拒绝；
只有回答实际提到月缴时才要求月缴金额。此检查只核对基础费率是否出现，
不能代替责任归属、加油包组合金额、所有数字与条件的人工检查。

| 文件 | 内容 |
|---|---|
| `exp3_heldout.py` | 护栏保留集 1：v1 写完后编写，发现问题后参与了 v2 调整 |
| `exp3_heldout2.py` | 护栏保留集 2：v2 调整前编写，调整期间未查看结果，只用于最终评估 |
| `evidence.py` | 实验 3 共用的证据构造（与检索切片格式一致） |
| `exp5_audio_v2.py` | 第八轮：真实音频打断扩大（40 打断 / 30 附和 / 24 回声 × 三种回声条件） |
| `exp4_e2e_v5.py` | 第八轮：端到端语音时延扩大（30 句 × 2 轮），结果 `exp4_e2e_v4_expand*.json` |
| `build_qa_v4.py` | 第八轮：150 道全新问答保留题（`data/qa_v4.json`）；运行 `exp2_qa_v2.py --data qa_v4.json --tag v4 --agent-steps 5` |
| `*_review.json` | 第八轮逐题 / 逐通复核结果（`exp2_qa_v2_v4_review.json`、`exp8_calls_v3_review.json`） |

第八轮整通电话：`python -c "import sys; sys.path.insert(0,'experiments'); import exp8_calls as m; m.main(rounds=3, tag='v3')"`（直接运行 `exp8_calls.py` 会覆盖第六轮的 `exp8_calls_v2.json`）。
