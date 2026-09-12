# Data generation

`datagen/pipeline.py`是唯一CLI主入口，位于`datagen/`顶层；生成器与审核器
分别放入独立子目录。

## 目录与产物

```text
datagen/
  pipeline.py                       # 统一CLI主入口
  generation/
    catalog_foundation.py           # 从32-world catalog批量生成Foundation/Profile
    foundation.py                   # 基础资产空骨架
    anchor_draft.py                 # Stage 1/2共享前置草案
    stage1_history.py               # Stage 1连续历史
    stage1_tests.py                 # Stage 1 QA与开放任务
    stage2_tests.py                 # Stage 2最小反事实分支
    stage1_stage2.py                # Stage 1+2完整生成编排
  audit/
    source_assets.py                # 基础资产、Anchor与Transition门禁
    benchmark_assets.py             # 已生成Stage 1/2资产审计
    validation.py                   # Stage 1/2数量与结构验证
    HUMAN_REVIEW_RULES.md           # Foundation/Profile/Anchor人工规则
  shared/
    io.py                           # YAML/JSONL读写
    reviewed_sources.py             # 只读取审核通过的来源资产
    history_projection.py           # Player/NPC可见历史投影
```

- 基础资产：`generation/foundation.py`生成`canon.yaml`、
  `environment.yaml`、角色卡等空骨架。
- Stage 1/2共享前置：`generation/anchor_draft.py`生成
  `drafts/<npc>/anchors.yaml`、`transitions.yaml`和生成报告。
- Stage 1历史：`generation/stage1_history.py`生成
  `frozen/<npc>/history.jsonl`。
- Stage 1测试：`generation/stage1_tests.py`生成`qa.jsonl`和
  `open_tasks.jsonl`。
- Stage 2测试：`generation/stage2_tests.py`生成
  `branches/<npc>/pairs.jsonl`及分支历史。
- Stage 3场景：当前没有datagen生成器，使用人工审核的
  `scenarios/*.yaml`。

`stage1_stage2.py`按顺序调用Stage 1和Stage 2生成器，并写
`generated_manifest.yaml`。它不生成Stage 3数据。

## 主入口

所有全局参数放在`generate`或`audit`之前。

按审核目录批量生成未批准的Foundation与Profile草案：

```bash
python3 -m datagen.pipeline generate catalog-foundation \
  --catalog configs/datagen/world_catalog.yaml \
  --registry configs/llm/registry.yaml \
  --model venus-gpt-5.5 \
  --workers 4
```

该命令跳过现有`world_001/002`，逐世界写入`world_003–032`。所有
`source_review.yaml`检查默认为`false`，必须按
[`audit/HUMAN_REVIEW_RULES.md`](audit/HUMAN_REVIEW_RULES.md)人工审核。

创建基础资产空骨架：

```bash
python3 -m datagen.pipeline \
  --world assets/world_003 \
  --character npc_one \
  --character npc_two \
  generate foundation \
  --world-id world_003 \
  --name example_world \
  --language zh
```

生成并审核Anchor草案：

```bash
python3 -m datagen.pipeline --world assets/world_002 --character yu_zecheng \
  audit foundation

python3 -m datagen.pipeline --world assets/world_002 --character yu_zecheng \
  generate anchor-draft \
  --registry configs/llm/registry.yaml \
  --model venus-gpt6-astra

python3 -m datagen.pipeline --world assets/world_002 --character yu_zecheng \
  audit anchors
```

生成Stage 1与Stage 2正式数据：

```bash
python3 -m datagen.pipeline --world assets/world_002 --character yu_zecheng \
  generate stage1-2 \
  --registry configs/llm/registry.yaml \
  --model venus-deepseek-v4-pro \
  --rounds 600 \
  --pairs 10 \
  --workers 12
```

上述三个LLM生成入口均可用`--model`切换注册模型。旧版
`--config configs/models/*.yaml`仍兼容，但不能与单独的`--model`混用。
注册模型的认证、能力、超时、重试和并发限制统一由
`configs/llm/registry.yaml`控制。

审核已生成的Stage 1/2数据：

```bash
python3 -m datagen.pipeline --world assets/world_002 --character yu_zecheng \
  audit outputs --rounds 600 --pairs 10
```

查看完整命令：

```bash
python3 -m datagen.pipeline --help
python3 -m datagen.pipeline --world assets/world_002 generate --help
python3 -m datagen.pipeline --world assets/world_002 audit --help
```

## 门禁边界

- LLM生成的Anchor/Transition只写入`drafts/`，不能直接作为正式数据。
- 人工确认后复制到`frozen/<npc>/`，并通过`audit anchors`。
- Stage 1/2生成前会再次检查基础资产和Anchor门禁。
- 历史质量失败时只保留工作报告，不写入正式资产。
- Stage 3场景只包含初始状态和contracts，不预生成测试轨迹。
