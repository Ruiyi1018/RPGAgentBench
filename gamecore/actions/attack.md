---
name: attack
category: world
description: 以造成伤害为目的对目标发起攻击。
parameters:
  target:
    type: string
    required: true
  method:
    type: string
    required: true
updates:
  - environment.health
handler: attack
---

# Attack

模型选择攻击目标和方式，GameCore根据状态与规则计算实际伤害。
