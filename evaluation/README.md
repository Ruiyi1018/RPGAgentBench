# Evaluation

Stage 3由两个通用运行时组件组成：

- `stage3_contracts.py`：执行`forbidden/required`状态转移，以及`invariant/precedence/must_follow`轨迹契约；
- `challenge_controller.py`：根据契约状态推进挑战、注入公开fixture事件，并向Player暴露当前唯一目标。

场景只能在YAML中声明状态路径、Action模式、触发条件和挑战顺序。Python实现不得包含world、NPC或scenario专属判断。

正式失败统一分为：

1. `illegal_transition`：当前条件不允许提交的Action或后继状态；
2. `missing_transition`：触发条件成立后，相关决策机会耗尽仍未进入要求状态；
3. `trajectory_conflict`：单步可执行，但完整轨迹违反不变量、顺序或承诺生命周期。

格式错误、API错误和无状态变化的重复Action单独报告，不计入角色完整性失败。
