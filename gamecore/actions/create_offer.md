---
name: create_offer
category: world
description: 创建尚未执行的赠送或交换提议。
parameters:
  recipient:
    type: string
    required: true
  offered_items:
    type: array
    required: true
  requested_items:
    type: array
    required: false
updates:
  - environment.offers
handler: create_offer
---

# Create Offer

该Action只创建Pending Offer，不改变物品所有权。
