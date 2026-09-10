# Prompts

当前在线Prompt：

```text
prompts/
├─ player/
│  ├─ normal.{zh,en}.j2
│  ├─ pressure.{zh,en}.j2
│  └─ tactics/
├─ npc/
│  └─ online.{zh,en}.j2
├─ checker/
│  └─ consistency.{zh,en}.j2
└─ datagen/
```

每个在线Agent及语言只使用一个模板文件。模板中的`{% user %}`仅标记
API消息角色边界：其上为稳定指令，其下为动态上下文；Prompt内容在同一文件
中维护，发送时仍保留System/User隔离。

Player与NPC模板只能通过`prompt_builder.py`读取白名单投影，不得直接加载`GameContext`、`EvaluationSpec`或`role_contract`。

- NPC可见：去除测试字段的角色卡、场景名称、当前观察、允许的Action，以及由状态访问条件决定的历史或状态。
- Player可见：NPC公开身份、场景、对应条件目标、公开观察和公开历史。
- Checker可见：完整角色卡、`role_contract`和截至当前窗口的轨迹；为窗口内每轮输出单一冲突判定、规则ID和证据。

`PromptBuilder.from_environment(...)`读取`environment.yaml`中的`language`并选择对应模板；当前支持`zh`和`en`。
