---
name: decide_claim
category: role
description: 对一个已存在的事实主张作出接受、拒绝或暂不确定的判断。
parameters:
  claim_id:
    type: string
    required: true
    meaning: 要判断的主张ID，必须从合法参数候选中复制。
  decision:
    type: string
    enum: [accept, reject, uncertain]
    required: true
    meaning: NPC对该主张形成的当前判断。
  evidence_event:
    type: string
    required: false
    meaning: 直接支持本次判断或证据依据更新的既有历史Event ID，不是行动动机说明；改变既有判断时必须填写。
updates:
  - runtime_state.claims
handler: decide_claim
---

# Decide Claim

只记录NPC对事实命题的判断，不代表执行任务、授权或采取世界行为。
