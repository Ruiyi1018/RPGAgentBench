---
name: create_artifact
category: world
description: 创建本轮产生并由NPC持有的普通便签、摘要、报告、记录或消息。
parameters:
  artifact_id:
    type: string
    required: true
    meaning: 本轮新建信息载体的唯一ID。
  kind:
    type: string
    enum: [note, summary, report, record, message, other]
    required: true
    meaning: 信息载体类型。
  content:
    type: string
    required: true
    meaning: 新建载体中实际写入的内容。
updates:
  - environment.artifacts
  - environment.inventories
handler: create_artifact
---

# Create Artifact

只创建由NPC持有的新信息载体；需要交付时另行使用 `transfer_item`。
