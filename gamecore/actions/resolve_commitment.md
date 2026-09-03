---
name: resolve_commitment
category: role
description: 改变一项已有承诺的状态。
parameters:
  commitment_id:
    type: string
    required: true
  resolution:
    type: string
    enum: [fulfilled, cancelled, violated]
    required: true
  reason_event:
    type: string
    required: true
updates:
  - runtime_state.commitments
handler: resolve_commitment
---

# Resolve Commitment

只能处理已存在且尚未结束的承诺，并必须引用支持该变化的历史事件。
