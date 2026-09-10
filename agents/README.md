# Agents

该目录只负责角色行为与Prompt构建；模型接口和供应商客户端位于顶层`llm/`。

当前实现：

- `player.py`：Normal与Pressure Player，只生成`query`；
- `npc.py`：生成`utterance`和最多两个Action；
- `prompt_builder.py`：从同一`GameContext`分别拼接Player与NPC prompt；
- `consistency_checker.py`：批量逐项判断NPC是否与角色契约或既有世界状态冲突；
- `prompts/`：Agent与数据生成模板。

NPC默认使用`history_only`，不接收结构化`RuntimeState`。NPC角色卡投影会移除`testable_boundaries`、`prohibitions`、秘密等级标签、`EvaluationSpec`和场景`action_targets`，只保留自然背景、自身经历、原则、目标与能力。Player可接收压力阶段目标但看不到Checker规则；Checker读取完整角色卡、`role_contract`和截至当前窗口的轨迹。Runner默认每5轮调用一次，初审报冲突时再调用一次独立确认。

Action参数候选只提供当前真实可执行的操作ID，例如当前Player主张、可访问资源和当前位置的相邻地点，不提供攻击目标或受保护事实清单。
