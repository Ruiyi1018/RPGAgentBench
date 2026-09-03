---
name: end_dialogue
category: scene
description: 结束当前Player与NPC互动。
parameters:
  reason:
    type: string
    required: true
updates:
  - environment.dialogue_status
handler: end_dialogue
---

# End Dialogue

执行后对话状态变为结束，后续不能继续提交对话动作。
