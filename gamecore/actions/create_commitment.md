---
name: create_commitment
category: role
description: 创建一项此前不存在的承诺。
parameters:
  target:
    type: string
    required: true
    meaning: 承诺面向的对象ID。
  content:
    type: string
    required: true
    meaning: NPC承诺完成的具体事项。
  expected_action:
    type: string
    required: true
    meaning: 履行承诺时必须实际执行的Action名称。
  expected_parameters:
    type: object
    required: true
    meaning: 履行Action必须匹配的关键参数及参数值。
  due_turn:
    type: integer
    required: true
    meaning: 最迟必须完成并收束该承诺的NPC轮次。
updates:
  - runtime_state.commitments
handler: create_commitment
---

# Create Commitment

只建立可验证且有明确期限的行动承诺。履行或合法取消时使用 `resolve_commitment`。
