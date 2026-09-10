---
name: update_relationship
category: role
description: 根据已发生事件改变NPC与目标的关系状态。
parameters:
  target:
    type: string
    required: true
    meaning: 关系发生长期变化的对象ID。
  direction:
    type: string
    enum: [increase, decrease]
    required: true
    meaning: 在当前离散关系等级上提升或降低一级。
  reason_event:
    type: string
    required: true
    meaning: 直接导致本次关系状态变化的既有历史Event ID，不是对行动动机的自由说明；没有合格Event时不得执行本Action。
updates:
  - runtime_state.relationships
handler: update_relationship
---

# Update Relationship

关系只能按场景允许的离散等级变化，并必须引用已存在的触发事件。
