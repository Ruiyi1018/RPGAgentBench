# RPG-AgentBench

RPG-AgentBench 是一个面向长程角色扮演 Agent 的可执行评测框架。它不只检查模型能否模仿角色语气，还关注模型能否在长历史、多轮交互和持续压力下：

- 正确理解角色经历过的事件与当前状态；
- 根据状态变化做出不同且可执行的决定；
- 维持身份、知识、关系、目标和承诺的一致性；
- 保持语言与实际 Action 一致；
- 在拒绝危险请求时仍然提供有价值的互动。

框架将自然语言角色扮演、确定性状态机和自动评测分离。Stage 3将各类角色冲突统一为一个可复核的失败事件，避免用互相重叠的标签重复计分。

## 评测协议

### Stage 1：冻结历史理解

模型读取角色卡与长程冻结历史，完成分层结构化 QA 和开放式角色回复任务。QA 覆盖 Profile、Temporal、Episodic、Local State、Checkpoint State、Long-range Final State 和 Multi-hop；其中 Episodic 直接要求恢复普通历史中后来再次影响判断的细节，避免600轮历史退化为只包裹Anchor的填充文本。

### Stage 2：最小分支决策

两条分支共享相同的长历史，仅在末尾保留一个最小可见差异。模型需要分别给出自然语言回复和正式 Action。

- Sensitivity Pair：关键状态改变时，决策也应改变；
- Invariance Pair：无关表述变化时，核心决策应保持稳定。

这一阶段用于判断模型是否真正依赖角色状态，而不是只套用固定人格模板。

### Stage 3：在线交互压力测试

Player Agent 与 NPC Agent 在 GameCore 中进行多轮交互：

- Normal：合法、合作式任务推进；
- Pressure：冒充授权、紧迫性施压、利益诱导、道德绑架和渐进式越界请求。

Stage 3 v8以通用`stage3_contract`作为唯一可执行评测对象。`Transition Contract Engine`确定性检测三类失败：当前状态不允许的非法转移、触发后在相关决策机会内仍未发生的缺失转移，以及违反顺序、承诺或状态持久性的轨迹冲突。`Challenge Controller`按契约生命周期注入公开事件并给Player分配当前挑战，不再按绝对轮次硬编码压力脚本。相同判断的无变化重复提交只记为`redundant_action`诊断，不终止轨迹。Consistency Checker默认每5轮批量检查台词、Action与实际状态变化是否语义一致，并对疑似冲突二次确认。

## 总体架构

```text
                         ┌─────────────────────────┐
                         │  Human-reviewed assets  │
                         │ canon / cards / anchors │
                         │ transitions / scenarios│
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
          Offline data generation             Online evaluation
                    │                                   │
       session planner + dialogue LLM                   │
                    │                                   │
        600-round executable history                    │
          ├── Stage 1 QA/Open Tasks                     │
          └── Stage 2 Branch Pairs                      │
                                                        │
 Player Agent ──query──> Prompt Builder ──> NPC Agent
                                            │ utterance
                                            │ + actions
                                            ▼
                                      GameCore Engine
                                            │
                                      GameContext
                                            │
                                  Consistency Checker
                                            │
                                       Metrics/Report
```

### 核心组件

- **GameContext**：运行时单一事实源，保存角色状态、环境状态和事件历史。
- **GameCore**：唯一状态写入者；校验 Action、原子执行状态变化并记录结果。
- **Prompt Builder**：按 Agent 身份投影可见信息，阻止隐藏状态和评测规则泄漏。
- **NPC Agent**：输出角色化自然语言与增量 Action。
- **Player Agent**：只输出自然语言请求，不直接操作 GameCore。
- **Consistency Checker**：只读取最近检查窗口，补查台词、Action与GameCore实际变化的语义一致性。
- **LLM Layer**：统一供应商配置、结构化输出重试和 OpenAI-Compatible/Venus 调用。
- **Data Generator**：将人工 Anchor 和 Transition 转换为连续 Session、长历史、QA 及分支对。

## 设计约束

1. Agent 不能直接修改 `GameContext`，所有状态变化必须经过 GameCore。
2. 同一轮 Action 作为原子事务执行；任一 Action 无效时，本轮状态变化不提交。
3. Player 不调用 Action；Stage 3固定向NPC提供全部13个Action，由NPC自行选择。
4. NPC 看到完整Action契约、合法参数候选和自己尚未结束的承诺，不看到完整内部状态空间。
5. `stage3_contract`、完整`challenge_plan`和未来事件不进入NPC或Player Prompt；Player只接收当前已激活挑战的公开目标，不读取约束判定条件。
6. 隐藏事件通过可见性字段隔离；不同 Agent 获得不同的历史投影。
7. 长历史由“状态机 → Session 规划 → 多轮轨迹 → 角色化实现”生成，不使用独立问答式填充轮次。
8. 生成中间文件默认写入系统临时目录并自动清理，不混入正式资产。

## 数据生成

数据流程从正典范围、世界、角色画像开始，依次经过Foundation机器审查、人工清单、Anchor草案、Anchor机器审查和人工清单。只有两层门禁都通过，才能生成正式数据。

每个正式角色由人工审核的`anchors.yaml`和`transitions.yaml`驱动。默认600轮协议包含30个长度不同的连续Session和30个Anchor。Anchor保持因果顺序，但非均匀分布在Session与前、中、后位置：部分Session无Anchor，少数Session含多个Anchor，部分影响延迟到后续Session才显现。协议要求：

- 后一轮承接上一轮；
- Anchor 后的状态变化映射为可观察行为；
- 普通细节在后文被重新引用并影响判断；
- Player 身份和权限保持稳定；
- 无 Schema 术语、模板套话、重复词和隐藏信息泄漏。

生成完成后自动构造：

```text
frozen/<npc_id>/history.jsonl       600轮Player/NPC历史
frozen/<npc_id>/qa.jsonl            50条分层QA
frozen/<npc_id>/open_tasks.jsonl    2条开放任务
branches/<npc_id>/pairs.jsonl       7组Sensitivity + 3组Invariance
branches/<npc_id>/<pair_id>/        两条物化分支历史
```

生成器会检查结构、GameCore可执行性、信息隔离、Anchor状态覆盖、语言质量、QA可测量性以及Pair最小性。完整规则与分阶段命令见`datagen/README.md`。

## 仓库结构

```text
assets/       世界、角色卡、Anchor、Transition、场景和生成数据
schemas/      GameContext与Agent结构化输出契约
gamecore/     状态容器、Action注册、执行引擎和确定性处理器
  actions/    13个职责互斥的正式Action Markdown契约
llm/          模型接口、配置加载、供应商适配与客户端工厂
agents/       NPC、Player、Checker和Prompt Builder
  prompts/    按Agent与语言组织的Prompt模板
datagen/      Session生成、QA、分支、审计与世界级生成入口
runners/      Pilot与完整Stage 1–3实验入口
evaluation/   指标与报告逻辑
configs/      模型和实验配置
tests/        GameCore、Agent、数据、Runner与LLM回归测试
runs/         实验输出与Checkpoint
```

单个世界的主要资产结构：

```text
assets/world_xxx/
  canon.yaml
  environment.yaml
  characters/<npc_id>.yaml
  frozen/<npc_id>/
    source_review.yaml
    anchors.yaml
    transitions.yaml
    history.jsonl
    qa.jsonl
    open_tasks.jsonl
  branches/<npc_id>/
  scenarios/*.yaml
```

当前主要开发世界为 `world_002`，正典来源是电视剧《潜伏》；评测角色包括余则成、王翠平、吴敬中、李涯、陆桥山和谢若林。

## 安装

需要 Python 3.10 或更高版本：

```bash
python3 -m pip install -e ".[test]"
python3 -m pytest
```

## 模型配置

默认数据生成配置为 `configs/models/venus_deepseek.yaml`，使用 Venus 内部部署的 `deepseek-v4-pro`。

在项目根目录创建 `.env`：

```bash
LLM_PROVIDER=venus
VENUS_API_KEY='your-token'
VENUS_MODEL=deepseek-v4-pro
```

Token 不应写入 YAML、命令参数或生成结果。其他 OpenAI-Compatible 模型可以通过独立配置文件接入。

## 常用命令

先检查世界、角色和Anchor来源资产：

```bash
python3 -m datagen.pipeline \
  --world assets/world_002 \
  --character yu_zecheng \
  audit sources
```

两层人工清单通过后，生成一个角色的完整离线数据：

```bash
python3 -m datagen.pipeline \
  --world assets/world_002 \
  --character yu_zecheng \
  generate stage1-2 \
  --config configs/models/venus_deepseek.yaml \
  --rounds 600 \
  --workers 6
```

生成整个世界：

```bash
python3 -m datagen.pipeline \
  --world assets/world_002 \
  generate stage1-2 \
  --config configs/models/venus_deepseek.yaml \
  --rounds 600 \
  --workers 6 \
  --replace
```

`--replace` 允许覆盖已存在角色的正式生成结果；只生成尚无数据的单个角色时不需要该参数。

运行不调用 API 的预检：

```bash
python3 -m runners.full_experiment \
  --world assets/world_002 \
  --character yu_zecheng \
  --scenario assets/world_002/scenarios/yu_zecheng_archive_request.yaml \
  --dry-run
```

运行完整 Stage 1–3 实验：

```bash
python3 -m runners.full_experiment \
  --world assets/world_002 \
  --character yu_zecheng \
  --scenario assets/world_002/scenarios/yu_zecheng_archive_request.yaml \
  --workers 6
```

低成本 Stage 1/2 API Smoke：

```bash
python3 -m runners.pilot \
  --world assets/world_002 \
  --character yu_zecheng
```

## 主要指标

- Stage 1：QA Exact/F1、证据定位、开放任务通过率；
- Stage 2：分支决策准确率、Sensitivity 与 Invariance 一致性；
- Stage 3 Integrity Failure Rate：分别报告`illegal_transition`、`missing_transition`、`trajectory_conflict`，并报告三者合并后的轨迹失败比例；
- TTFF：首次确认角色冲突的轮次；
- Survival：第10/20/30/40轮仍未发生确认冲突的比例；只统计实际达到该检查点或此前已经失败的轨迹；
- Challenge Coverage与Contract Status：诊断状态转移测试族是否完成，以及每份契约处于`inactive/active/satisfied/violated`的哪一阶段；覆盖不足的无失败轨迹标记为`insufficient_coverage`，不解释为正式通过；
- Action诊断：分别报告非空Action轮比例、GameCore语义拒绝率、`redundant_action`次数和不同Action覆盖率。

所有实验结果和 Checkpoint 写入 `runs/`。正式资产不会在评测过程中被修改。
