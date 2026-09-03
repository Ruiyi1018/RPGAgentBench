# Assets

该目录只保存数据，不放置Python代码。

每个正式世界使用相同结构：

```text
world_xxx/
├─ environment.yaml
├─ canon.yaml
├─ anchor_pool.yaml
├─ characters/
├─ frozen/
├─ branches/
└─ scenarios/
```

`environment.yaml`只定义地点连接和物品类型。角色背景与原则写入角色卡；初始位置、物品归属、关系、承诺和任务状态写入场景的`GameContext`。

`anchor_pool.yaml`保存正典事件，`frozen/{npc_id}/anchors.yaml`直接保存该NPC使用的正典锚点和原创可控锚点，每名NPC合计30个；具体配比取决于该角色在正典中的有效事件数量。
