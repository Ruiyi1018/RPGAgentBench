---
name: use_item
category: world
description: 非攻击性地使用NPC可用的物品。
parameters:
  item:
    type: string
    required: true
    meaning: NPC当前持有且本轮实际使用的物品ID。
  target:
    type: string
    required: true
    meaning: 该物品实际作用的角色或对象ID。
updates:
  - environment.inventories
  - environment.health
handler: use_item
---

# Use Item

物品必须由NPC持有且支持对应用途。以造成伤害为目的时必须使用 `attack`。
