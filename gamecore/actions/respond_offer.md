---
name: respond_offer
category: world
description: 接受或拒绝一项已有的赠送或交换提议。
parameters:
  offer_id:
    type: string
    required: true
  decision:
    type: string
    enum: [accept, reject]
    required: true
updates:
  - environment.offers
  - environment.inventories
handler: respond_offer
---

# Respond Offer

接受时由GameCore验证双方物品并原子执行转移；拒绝时只关闭Offer。
