---
name: decide_access
category: world
description: 授予、拒绝或撤销目标对地点或物品的访问权限。
parameters:
  subject:
    type: string
    required: true
  resource:
    type: string
    required: true
  decision:
    type: string
    enum: [grant, deny, revoke]
    required: true
updates:
  - environment.access
handler: decide_access
---

# Decide Access

NPC必须拥有相应权限；拒绝也会作为本轮明确决定写入历史。
