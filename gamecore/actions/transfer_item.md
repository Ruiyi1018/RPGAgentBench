---
name: transfer_item
category: world
description: 将NPC当前持有的物品交给角色或放到地点。
parameters:
  item:
    type: string
    required: true
    meaning: 要转移的物品ID，必须由NPC当前持有。
  source:
    type: string
    required: true
    meaning: 来源角色ID，必须是当前NPC。
  destination:
    type: string
    required: true
    meaning: 接收角色或地点ID，必须从合法参数候选中复制。
updates:
  - environment.inventories
handler: transfer_item
---

# Transfer Item

只执行NPC已经决定的实际交付，不表示提议、讨论或尚未完成的交换。
