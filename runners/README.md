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

默认执行50道分层QA、2个开放任务、10个Pair，以及Normal/Pressure各40轮、3个seed。所有Provider默认将QA拆为2个25题批次。GameCore每轮即时检查Action和确定性角色规则；固定Checker默认每5轮批量补查语言层冲突，初审发现冲突时再批量确认一次。预检报告正常QA批量回答时602次、把QA部分回答恢复计入后650次的保守逻辑调用上界；结构化输出内部重试不计为新的逻辑调用。若模型只返回QA批次中的部分答案，运行器会立即保存有效答案并仅补问遗漏的`qa_id`。可用`--qa-batch-size`、`--checker-interval`、`--seeds`、`--turns`、`--player-model`和`--checker-model`调整。正式比较应让Player和Checker使用不同于被测NPC的模型。

Stage 3 v8使用场景声明的`stage3_contract`和`challenge_plan`，Runner中不包含人物、地点、物品或绝对轮次特判。Contract Engine报告`illegal_transition`、`missing_transition`和`trajectory_conflict`；Challenge Controller根据契约结果或机会预算切换测试目标。GameCore命中的确定性失败立即终止；无状态变化的重复Action只记诊断；语言层失败在当前5轮窗口结束时确认。未失败但没有完成挑战计划的轨迹记为`insufficient_coverage`，不作为正式通过。格式错误记为`invalid`并从失败率和生存率分母中排除。
