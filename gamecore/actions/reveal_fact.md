---
name: reveal_fact
category: role
description: 向Player公开NPC已知的事实或秘密。
parameters:
  fact_id:
    type: string
    required: true
    meaning: 本轮实际披露的已知事实ID，必须从合法参数候选中复制。
  recipient:
    type: string
    required: true
    meaning: 获得该事实的接收者ID。
updates:
  - runtime_state.disclosures
handler: reveal_fact
---

# Reveal Fact

只能公开NPC当前已知的事实；公开内容会作为本轮动作事件写入历史。
