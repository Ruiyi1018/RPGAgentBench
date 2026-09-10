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
│  └─ npc_id/
│     ├─ initial_context.yaml（新角色可选）
│     ├─ anchors.yaml
│     ├─ transitions.yaml
│     └─ source_review.yaml
├─ drafts/
├─ branches/
└─ scenarios/
```

`environment.yaml`只定义地点连接和物品类型。角色背景与原则写入角色卡；初始位置、物品归属、关系、承诺和任务状态写入场景的`GameContext`。

`anchor_pool.yaml`保存正典事件，`frozen/{npc_id}/anchors.yaml`直接保存该NPC使用的正典锚点和原创可控锚点，每名NPC合计30个；具体配比取决于该角色在正典中的有效事件数量。

`source_review.yaml`是人工门禁，只保存布尔检查结果，不保存评审者的思考过程。`drafts/`中的LLM草案不能被Runner或正式数据生成直接使用；人工确认并复制到`frozen/`后，还必须通过`python3 -m datagen.pipeline --world <world> --character <npc> audit anchors`。

角色画像、世界和Anchor的字段约束及人工规则见`datagen/README.md`。任何新世界都必须遵守相同目录和审查流程，不得在Python中添加作品或角色专用分支。
