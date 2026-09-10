---
name: resolve_commitment
category: role
description: 改变一项已有承诺的状态。
parameters:
  commitment_id:
    type: string
    required: true
    meaning: 要收束的有效承诺ID，必须从合法参数候选中复制。
  resolution:
    type: string
    enum: [fulfilled, cancelled]
    required: true
    meaning: fulfilled表示已经完成承诺；cancelled表示因新事件合法取消。
  reason_event:
    type: string
    required: true
    meaning: fulfilled时必须引用与承诺expected_action及关键参数匹配的成功Action Event；cancelled时引用承诺建立后支持取消的既有Event。期限或催办Event不能证明履行。
updates:
  - runtime_state.commitments
handler: resolve_commitment
---

# Resolve Commitment

只能处理已存在且尚未结束的承诺。履行时，引用事件必须由承诺要求的Action产生。
