# Actions

Action只表示会改变`GameContext`的操作。回答、拒绝、解释、谈判、威胁和情绪表达不属于Action。

每个Action使用一个独立Markdown：

- YAML front matter：名称、类别、参数、更新路径和处理器；
- 正文：用途、边界和与相近Action的区别。

空`actions`表示本轮没有状态变更。每个场景只向NPC提供相关的4–6个Action。
