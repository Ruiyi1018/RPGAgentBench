# Prompts

当前在线Prompt：

```text
prompts/
├─ player/
│  ├─ normal_system.{zh,en}.j2
│  ├─ pressure_system.{zh,en}.j2
│  ├─ online_user.{zh,en}.j2
│  └─ tactics/
├─ npc/
│  ├─ online_system.{zh,en}.j2
│  └─ online_user.{zh,en}.j2
├─ checker/
│  ├─ consistency_system.{zh,en}.j2
│  └─ consistency_user.{zh,en}.j2
└─ datagen/
```

Player与NPC模板只能通过`prompt_builder.py`读取白名单投影，不得直接加载`GameContext`或`EvaluationSpec`。

- NPC可见：去除测试字段的角色卡、场景名称、当前观察、允许的Action，以及由状态访问条件决定的历史或状态。
- Player可见：NPC公开身份、场景、对应条件目标、公开观察和公开历史。
- Checker可见：完整角色卡、EvaluationSpec和完整episode轨迹。

`PromptBuilder.from_environment(...)`读取`environment.yaml`中的`language`并选择对应模板；当前支持`zh`和`en`。
