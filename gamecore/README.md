# GameCore

GameCore是零LLM依赖的唯一状态写入者。

当前实现：

- `context.py`：加载和保存单一`GameContext`；
- `actions/*.md`：13个正式Action的参数、状态更新和使用边界；
- `action_registry.py`：读取并校验`gamecore/actions/*.md`的YAML头部；
- `engine.py`：验证Action、调用处理器、更新状态并追加历史；
- `fixture_engine.py`：原子执行人工审核的离线世界事件，生成可审计的前后状态与delta；
- `handlers.py`：13个Action对应的确定性状态更新函数；
- `errors.py`：稳定的验证错误类型和错误码。

Agent不得直接修改`GameContext`。

同一轮的Action按原子事务执行：任一Action无效，本轮所有状态变更均不提交，但失败尝试会记录为`decision_violation`。GameCore不会修复语义错误，也不会向NPC返回具体角色规则或正确Action。
