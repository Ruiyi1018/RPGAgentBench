---
name: create_commitment
category: role
description: 创建一项此前不存在的承诺。
parameters:
  target:
    type: string
    required: true
  content:
    type: string
    required: true
  condition:
    type: string
    required: false
updates:
  - runtime_state.commitments
handler: create_commitment
---

# Create Commitment

仅用于建立新承诺。完成、取消或违反已有承诺时使用 `resolve_commitment`。
