---
name: reveal_fact
category: role
description: 向Player公开NPC已知的事实或秘密。
parameters:
  fact_id:
    type: string
    required: true
  recipient:
    type: string
    required: true
updates:
  - runtime_state.disclosures
handler: reveal_fact
---

# Reveal Fact

只能公开NPC当前已知的事实；公开内容会作为本轮动作事件写入历史。
