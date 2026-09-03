---
name: transfer_item
category: world
description: 直接给予、拿取、放下或归还物品。
parameters:
  item:
    type: string
    required: true
  source:
    type: string
    required: true
  destination:
    type: string
    required: true
updates:
  - environment.inventories
handler: transfer_item
---

# Transfer Item

物品必须由来源持有或位于来源地点。需要对方先同意的赠送或交换应使用Offer流程。
