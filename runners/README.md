# Runners

三个正式运行入口：

- `frozen.py`：Stage 1冻结轨迹诊断；
- `branch.py`：Stage 2分支历史决策；
- `online.py`：Stage 3 Normal/Pressure在线互动。

所有运行轨迹写入`runs/`，不得修改测试资产。

当前已实现`pilot.py`，用于World 002的Stage 1/2小规模API实验。先运行不调用API的预检：

```bash
python3 -m runners.pilot \
  --world assets/world_002 \
  --character yu_zecheng \
  --dry-run
```

预检通过后，将Venus应用组完整Token写入项目根目录`.env`：

```bash
LLM_PROVIDER=venus
VENUS_API_KEY='...'
VENUS_MODEL=deepseek-v4-pro
```

Runner会按照`configs/models/venus_deepseek.yaml`自动加载`.env`，
随后直接运行：

```bash
python3 -m runners.pilot \
  --world assets/world_002 \
  --character yu_zecheng
```

默认固定使用Venus内部`deepseek-v4-pro`、10道QA和2个Pair，共5次
API调用。可通过`--config`、`--qa-limit`和`--pair-limit`调整；
Venus路由不接受其他模型名。Token不会写入YAML、输出或命令参数。

完整Stage 1–3协议使用`full_experiment.py`：

```bash
python3 -m runners.full_experiment \
  --world assets/world_002 \
  --character yu_zecheng \
  --scenario assets/world_002/scenarios/yu_zecheng_archive_request.yaml \
  --dry-run
```

默认执行50道分层QA、2个开放任务、10个Pair，以及Normal/Pressure各40轮、3个seed，共512次基础调用。可用`--seeds`、`--turns`、`--player-model`和`--checker-model`调整。正式比较应让Player和Checker使用不同于被测NPC的模型。

Stage 3分别报告角色、执行、格式、grounding和效用首次失败轮次，并给出第10/20/30/40轮生存率；`TTFF-any`不再替代角色失败指标。
