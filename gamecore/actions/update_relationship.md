---
name: update_relationship
category: role
description: 根据已发生事件改变NPC与目标的关系状态。
parameters:
  target:
    type: string
    required: true
  direction:
    type: string
    enum: [increase, decrease, maintain]
    required: true
  reason_event:
    type: string
    required: true
updates:
  - runtime_state.relationships
handler: update_relationship
---

# Update Relationship

关系只能按场景允许的离散等级变化，并必须引用已存在的触发事件。
