# 保险复杂咨询语音 Agent

基于尊享e生百万医疗2026材料的复杂咨询演示项目。当前入口使用`EvidencePipeline`与`CallSession`：完整证据一次生成、会话条件更正、持续咨询待办、来源引用与后置人工请求。旧`Pipeline`和实验脚本保留用于历史对照。

## 当前版本

核心冻结标签：`v0.1.0-core-20261014`。源文件及清洗数据的SHA-256清单见`releases/core-2026-10-14.json`；直接依赖版本见`requirements-core.lock`。冻结是可复现的工程快照，不表示赛题全部通过。

当前完整批次：复杂/复合题26/30（86.7%），整体正确处理133/143（93.0%）；复杂完整口播、附和/回声误停、来源支持和安全人工路由仍有缺口。后续规划修复通过124项单元测试和5个真实Qwen规划用例，完整语音待复验。

## 项目结构

| 目录 | 内容 |
|---|---|
| `src/agent` | 当前业务链路及历史对照链路 |
| `data/clean` | 清洗条款、保障责任、费率与原文定位 |
| `experiments` | 开发与验证脚本；结果按批次保留，不覆盖失败记录 |
| `tests` | 条件、任务、来源、取消与播放回执的程序回归 |
| `docs` | 方案、实验记录、指标口径与未达标项 |
| `releases` | 冻结快照与来源清单 |

## 本地检查

```powershell
python -m pip install -r requirements-core.lock
python -m unittest discover -s tests -q
```

云服务凭据从本地`.env`与可选`.env.deepseek`读取，文件已忽略。没有凭据也可运行程序单元测试；真实模型、ASR和TTS必须配置对应服务。冻结清单仅记录直接依赖，不声称完整锁定全部传递依赖或外部模型版本。

相关记录：[当前指标](docs/24-当前指标与验收口径.md)、[后续任务](docs/07-TODO.md)、[最新规划修复](docs/29-当前问项与历史待办隔离.md)。

## 网页演示

答辩演示标签：`v0.2.0-defense-20261014`，在冻结核心之外提供网页服务。配置DeepSeek后默认实时自由提问；未配置时使用离线回放，并提示缺失。Qwen仅用于主动选择的对照。

```powershell
.\start-demo.ps1
```

打开`http://127.0.0.1:8080`。已有依赖时可直接运行`python scripts/run_demo.py`；当前Windows TLS异常环境使用`--native-https`。支持现场语音新问题、说完自动发送、0.75–2倍播放、连续咨询、条件更正、实时阶段耗时、完整原文溯源、逐轮生成/已播记录、JSON导出和3+7+5分钟演练计时。

可选容器启动：`docker compose up --build -d`。凭据使用`.env.demo.example`对应的本地`.env.demo`，不进入镜像或Git。完整步骤、已测范围及容器未实测说明见[Web服务与冻结版本](docs/30-Web演示服务与冻结版本.md)。

验证冻结核心：`python scripts/verify_freeze.py`。答辩版142项测试通过，业务准确率和语音指标不因包装自动提升。配置检查：`python scripts/check_demo.py`。打包：`python scripts/package_submission.py`。

陈为个人参赛材料：[3分钟讲稿、7分钟演示与答辩准备](docs/31-陈为个人参赛答辩与提交说明.md)、[实现与验证口径](docs/32-答辩演示版实现与验证.md)。资料包包含原始结果和复现步骤，不包含服务密钥。
