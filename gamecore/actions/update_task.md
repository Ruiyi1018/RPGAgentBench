---
name: update_task
category: world
description: 更新一个已存在角色目标或场景任务的执行状态。
parameters:
  task_id:
    type: string
    required: true
    meaning: 要更新的目标或任务ID，必须从合法参数候选中复制。
  status:
    type: string
    enum: [active, completed, blocked, cancelled]
    required: true
    meaning: 该目标或任务的新状态。
  reason_event:
    type: string
    required: true
    meaning: 直接支持本次任务状态变化的既有历史Event ID，不是对行动动机的自由说明；没有合格Event时不得执行本Action。
updates:
  - runtime_state.goals
handler: update_task
---

# Update Task

统一记录角色目标和场景任务的状态变化，不再修改全局单一task_status。
