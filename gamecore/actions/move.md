---
name: move
category: world
description: 将NPC移动到与当前位置相连的地点。
parameters:
  destination:
    type: string
    required: true
    meaning: NPC本轮实际到达的相邻地点ID，必须从合法参数候选中复制。
updates:
  - environment.locations
handler: move
---

# Move

目标地点必须存在于 `environment.yaml`，并与NPC当前位置直接相连。
