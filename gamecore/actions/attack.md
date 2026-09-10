---
name: attack
category: world
description: 以造成伤害为目的对目标发起攻击。
parameters:
  target:
    type: string
    required: true
    meaning: 本轮实际遭受攻击的角色ID。
  method:
    type: string
    required: true
    meaning: 世界规则允许且NPC具备条件使用的攻击方式ID。
updates:
  - environment.health
handler: attack
---

# Attack

模型选择攻击目标和方式，GameCore根据状态与规则计算实际伤害。
