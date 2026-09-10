# Actions

Action只表示会改变`GameContext`的操作。回答、拒绝、解释、谈判、威胁和情绪表达不属于Action。

每个Action使用一个独立Markdown：

- YAML front matter：名称、类别、参数、更新路径和处理器；
- 正文：用途、边界和与相近Action的区别。

空`actions`表示本轮没有状态变更。Stage 3固定向NPC提供全部13个Action，由NPC自行选择；合法参数候选只约束ID和世界前置条件，不替NPC决策。

`create_artifact`只创建由NPC持有的信息载体，交付时另用`transfer_item`；`update_task`统一记录角色目标与场景任务；`resolve_commitment(fulfilled)`记录承诺履行。
