---
name: accept_claim
category: role
description: 接受或拒绝Player在历史中提出的一项事实主张。
parameters:
  claim_id:
    type: string
    required: true
  decision:
    type: string
    enum: [accept, reject, uncertain]
    required: true
  evidence_event:
    type: string
    required: false
updates:
  - runtime_state.claims
handler: accept_claim
---

# Accept Claim

Player的话语不会自动成为事实；该Action只记录NPC对某项主张的接受状态。
