# Data Generation

离线数据生成，不参与正式评测运行。

当前实现：

- `history_sources.py`：读取人工审核的Anchor与Transition；
- `llm_history.py`：规划连续session，并分块生成Player/NPC对话；
- `projection.py`：只向模型投影消息与其可见的事件，隔离答案和内部状态；
- `executable_qa.py`：生成Profile、Temporal、Local、Checkpoint、Long-range Final和Multi-hop六层QA；
- `executable_branches.py`：在完整历史后追加最小反事实Probe；
- `audit.py`：审计可执行性、可见性、语言重复、QA和Pair；
- `validate.py`：检查轮数、消息、锚点、QA证据、Pair配比和分支最小性；
- `generate_world.py`：世界级离线生成入口。

在线场景由`assets/<world>/scenarios/`定义，`EvaluationSpec`只提供给Checker和确定性审计，不进入NPC Prompt。

World 002生成命令：

```bash
python3 -m datagen.generate_world \
  --world assets/world_002 \
  --rounds 600 \
  --pairs 10 \
  --workers 12
```

只生成已修复的余则成数据：

```bash
python3 -m datagen.generate_world \
  --world assets/world_002 \
  --character yu_zecheng \
  --workers 12
```

每名NPC生成：

- `frozen/{npc_id}/history.jsonl`：600轮、1,200条消息；
- `frozen/{npc_id}/qa.jsonl`：50个结构化QA；
- `frozen/{npc_id}/open_tasks.jsonl`：2个开放回复任务；
- `branches/{npc_id}/pairs.jsonl`：10个Pair定义。

默认配置为`configs/models/venus_deepseek.yaml`，规划、对话生成和质量
审计均通过Venus调用内部`deepseek-v4-pro`。Stage 2两条分支共享完整
历史，只在末尾Probe事件上保留一处可见差异。
