---
name: decide_access
category: world
description: 授予、拒绝或撤销目标对地点或物品的访问权限。
parameters:
  subject:
    type: string
    required: true
    meaning: 权限决定所针对的角色ID。
  resource:
    type: string
    required: true
    meaning: 被控制的地点、物品或资源ID。
  decision:
    type: string
    enum: [grant, deny, revoke]
    required: true
    meaning: 授予、明确拒绝或撤销该权限。
  reason_event:
    type: string
    required: true
    meaning: 直接授权或支持本次权限决定的既有历史Event ID，不是对行动动机的自由说明；没有合格Event时不得执行本Action。
updates:
  - environment.access
handler: decide_access
---

# Decide Access

NPC必须拥有相应权限；拒绝也会作为本轮明确决定写入历史。
