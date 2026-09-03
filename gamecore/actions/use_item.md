---
name: use_item
category: world
description: 非攻击性地使用NPC可用的物品。
parameters:
  item:
    type: string
    required: true
  target:
    type: string
    required: true
updates:
  - environment.inventories
  - environment.health
handler: use_item
---

# Use Item

物品必须由NPC持有且支持对应用途。以造成伤害为目的时必须使用 `attack`。
