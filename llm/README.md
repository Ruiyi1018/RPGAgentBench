# LLM 模型接入、切换与调试指南

`llm/`是RPG-AgentBench唯一允许了解模型平台和远端模型差异的模块。
Agent、DataGen和Runner只依赖统一的`LLMClient.generate()`接口，不应出现
Venus、DashScope或具体模型名称的条件分支。

本文覆盖：

1. 当前模型分别适合做什么；
2. Stage 1-3如何分配模型；
3. 如何切换单个模型和批量运行模型矩阵；
4. 如何调试连通性、JSON、长上下文、输出截断和并发；
5. 如何新增、修改或删除模型。

## 目录与调用链

```text
Runner / DataGen / Agent
        │
        ▼
generate_structured()
        │ JSON Schema、解析、格式修复
        ▼
LLMClient.generate()
        │
        ├── VenusClient ───────────> Venus /chat/completions
        └── OpenAICompatibleClient > DashScope或其他兼容端点
```

主要文件：

- `base.py`：`GenerationConfig`、模型能力、统一Client协议、JSON解析与修复；
- `config.py`：兼容旧版单模型YAML和`.env`加载；
- `registry.py`：加载backend、model profile和角色分配；
- `factory.py`：根据backend创建具体客户端；
- `venus.py`：Venus请求构造、有限重试、usage与截断检查；
- `openai_compatible.py`：DashScope及OpenAI-compatible同步客户端；
- `rate_limit.py`：同进程、同远端模型共享的并发闸门；
- `../configs/llm/registry.yaml`：正式模型注册表；
- `../runners/model_smoke.py`：真实API冒烟测试；
- `../runners/model_sweep.py`：Stage 1-3模型矩阵入口。

## 实验角色

完整实验不直接按“NPC模型”“Judge模型”拼参数，而是使用三个稳定角色：

- `candidate`
  - Stage 1：QA和Open Task候选回答；
  - Stage 2：分支决策；
  - Stage 3：被测NPC。
- `player`
  - 只负责Stage 3的Player；
  - 主实验中固定，敏感性实验时才更换。
- `evaluator`
  - Stage 1：Open Task Auditor/Judge；
  - Stage 3：Consistency Checker；
  - 主实验中固定，敏感性实验时才更换。

即使三个角色选择同一远端模型，也会创建相互独立的客户端实例，避免usage、
request状态和格式重试记录互相覆盖。

## 当前模型与建议用途

下列“建议用途”是运行策略，不代表模型质量排名。模型能力必须由正式
Stage 1-3结果决定。

### Venus DeepSeek V4 Pro

注册名：`venus-deepseek-v4-pro`

建议用途：

- 默认基线candidate；
- 固定Player或Evaluator；
- 需要较稳定复杂结构化输出的实验。

特点：

- 使用`max_tokens`；
- 支持temperature、top_p、seed和JSON Schema；
- 默认最多4个并发请求；
- 真实Stage 1长Prompt冒烟可完整返回25题。

注意：用未经投影的12万字符原始JSONL做合成长上下文压力测试时，曾出现
输出预算被空白/隐藏推理耗尽。正式实验已启用token预算耗尽检查，不会把
这种结果计入评分。

### Venus DeepSeek V4 Flash

注册名：`venus-deepseek-v4-flash`

建议用途：

- 快速candidate基线；
- Pilot、开发期回归和大批量初筛；
- Player/Evaluator敏感性实验。

特点：

- 接口与Pro相同；
- 默认最多6个并发请求；
- 延迟通常低于Pro。

真实Stage 1的25题冒烟曾单次返回21题。完整Runner会保存有效答案并只补问
缺失`qa_id`，不会丢弃整个批次。

### Venus GPT 5.5

注册名：`venus-gpt-5.5`

建议用途：

- 正式candidate对比；
- 长上下文模型组；
- 固定Evaluator或Player的敏感性对照。

特点：

- 使用`max_completion_tokens`；
- 不发送temperature、top_p和seed；
- 支持严格JSON Schema；
- 默认最多2个并发请求；
- 已通过12万字符首、中、尾marker验证。

### Venus GPT 5.4 Mini

注册名：`venus-gpt-5.4-mini`

建议用途：

- 较快的GPT系candidate；
- Player敏感性实验；
- 低成本接口回归。

特点：

- 使用`max_completion_tokens`；
- 不发送旧式采样参数；
- 支持严格JSON Schema；
- 默认最多3个并发请求；
- 已通过12万字符marker验证。

### Venus GPT 6 Astra

注册名：`venus-gpt6-astra`

建议用途：

- 强模型candidate；
- Anchor/Foundation等高难度草案生成；
- 长上下文和复杂结构化输出对照。

特点：

- 使用`max_completion_tokens`；
- 不发送temperature、top_p和seed；
- 默认输出预算8192；
- 默认最多2个并发请求；
- 已通过严格JSON Schema和12万字符marker验证。

旧版Anchor配置`configs/models/venus_gpt6_astra_anchor.yaml`仍可使用。

### Venus Qwen 3.8 Flash Next

注册名：`venus-qwen3.8-flash-next`

建议用途：

- 快速candidate；
- 与DashScope直连Qwen比较平台差异；
- Player敏感性实验。

特点：

- 使用`max_tokens`；
- 支持采样参数、seed和严格JSON Schema；
- 默认最多4个并发请求；
- 已通过12万字符marker验证。

长输出压力测试中该部署能明确返回`finish_reason=length`，客户端会将其判为
截断失败，不会尝试解析半截JSON。

### Venus Kimi K2.6

注册名：`venus-kimi-k2.6`

建议用途：

- 长上下文candidate对照；
- 长轨迹理解敏感性实验。

特点：

- 使用`max_tokens`；
- 支持严格JSON Schema；
- 已通过12万字符marker验证。

注意：较长输出压力测试曾超过300秒。为避免一个模型拖住整个Sweep，当前
单次timeout为120秒，最多尝试2次，默认最多3并发。正式大跑前应先执行目标
任务的单模型smoke。

### Venus Gemini 3.8 Flash

注册名：`venus-gemini-3.8-flash`

建议用途：

- 快速Gemini系candidate；
- Player敏感性实验；
- 与Gemini Pro形成同系列对照。

特点：

- 使用`max_tokens`；
- 不发送seed；
- 支持严格JSON Schema；
- 默认最多4并发；
- 已通过12万字符marker验证。

Venus返回的finish reason可能是大写`STOP`，客户端会统一转小写后判断。

### Venus Gemini 3.1 Pro

注册名：`venus-gemini-3.1-pro`

建议用途：

- 强模型candidate；
- Evaluator敏感性实验；
- 长上下文复杂判断。

特点：

- 不发送seed；
- 支持严格JSON Schema；
- 默认最多2并发；
- 已通过12万字符marker验证。

该模型可能返回`MAX_TOKENS`，客户端会识别并拒绝截断输出。

### DashScope直连Qwen

注册名：

- `dashscope-qwen37-plus`
- `dashscope-qwen-plus`

用途：

- 验证同类模型直连外部平台时的差异；
- Venus与DashScope平台敏感性实验；
- Venus不可用时的显式替代实验。

它们使用`OpenAICompatibleClient`，需要`DASHSCOPE_API_KEY`。系统不会在
Venus失败时静默回退到DashScope，防止实验后端在不知情时发生变化。

## 密钥配置

密钥只放在项目根目录`.env`，不要写入YAML、命令或运行结果。

Venus支持两种方式。

方式一，完整Token：

```bash
VENUS_API_KEY='完整代理Token'
```

方式二，Secret ID加应用组后缀：

```bash
ENV_VENUS_OPENAPI_SECRET_ID='Secret ID'
```

`configs/llm/registry.yaml`中的Venus backend已配置：

```yaml
api_key_env: ENV_VENUS_OPENAPI_SECRET_ID
api_key_suffix: "@5784"
```

若同时设置，`VENUS_API_KEY`优先。Token不会写入smoke报告或实验输出。

DashScope：

```bash
DASHSCOPE_API_KEY='DashScope API Key'
```

## 注册表结构

Backend描述“怎么调用平台”：

```yaml
backends:
  venus:
    type: venus
    base_url: https://v2.open.venus.woa.com/llmproxy
    api_key_env: ENV_VENUS_OPENAPI_SECRET_ID
    api_key_suffix: "@5784"
    timeout_seconds: 300
    max_attempts: 3
    max_concurrency: 8
```

Model描述“平台上的哪个部署具有什么能力”：

```yaml
models:
  venus-gpt-5.5:
    backend: venus
    remote_model: gpt-5.5
    max_concurrency: 2
    capabilities:
      token_parameter: max_completion_tokens
      temperature: false
      top_p: false
      seed: false
      structured_output: json_schema
    defaults:
      max_tokens: 8192
      max_format_retries: 2
```

稳定注册名用于实验配置和目录名；`remote_model`才是发给API的模型ID。

## 数据生成时切换模型

DataGen的三个LLM入口均支持新注册表：

- `catalog-foundation`：生成World Foundation和角色Profile草案；
- `anchor-draft`：生成Anchor与Transition审核草案；
- `stage1-2`：生成连续历史，再确定性构造QA、Open Task和分支Pair。

Foundation/Profile示例：

```bash
python3 -m datagen.pipeline generate catalog-foundation \
  --catalog configs/datagen/world_catalog.yaml \
  --registry configs/llm/registry.yaml \
  --model venus-gpt-5.5 \
  --workers 4
```

Anchor草案建议优先使用较强且输出预算充足的模型，例如GPT 6 Astra：

```bash
python3 -m datagen.pipeline \
  --world assets/world_002 \
  --character yu_zecheng \
  generate anchor-draft \
  --registry configs/llm/registry.yaml \
  --model venus-gpt6-astra
```

生成Stage 1/2正式数据：

```bash
python3 -m datagen.pipeline \
  --world assets/world_002 \
  --character yu_zecheng \
  generate stage1-2 \
  --registry configs/llm/registry.yaml \
  --model venus-deepseek-v4-pro \
  --rounds 600 \
  --pairs 10 \
  --workers 6
```

这里的`--model`是单一“数据生成模型”，不使用candidate/player/evaluator
角色。Stage 1历史生成会并行创建Session计划和对话；所有请求仍受注册表中
该模型的`max_concurrency`限制。

数据生成模型的选择原则：

- Foundation和Anchor草案：优先强模型与较大输出预算；
- 600轮历史批量生成：优先结构稳定、延迟可接受的模型；
- 调试生成流程：可先用Flash/Mini模型和较小`--rounds`；
- 正式数据一旦通过人工审核，不应因切换被测candidate而重新生成。

旧版命令仍兼容：

```bash
--config configs/models/venus_deepseek.yaml
```

`--config`与`--registry/--model`是两种模型来源。使用注册表时必须同时提供
`--registry`和`--model`；不要只提供其中一个。

## 切换单次Stage 1-3实验模型

先只做预检，不调用API：

```bash
python3 -m runners.full_experiment \
  --world assets/world_002 \
  --character yu_zecheng \
  --scenario assets/world_002/scenarios/yu_zecheng_archive_request.yaml \
  --registry configs/llm/registry.yaml \
  --candidate-model venus-gpt-5.5 \
  --player-model venus-deepseek-v4-pro \
  --evaluator-model venus-deepseek-v4-pro \
  --dry-run
```

确认模型、调用上界和Prompt长度后，删除`--dry-run`正式运行。

主实验只需要修改`--candidate-model`。例如：

```bash
--candidate-model venus-gemini-3.1-pro
```

敏感性实验可以单独修改：

```bash
--player-model venus-gpt-5.4-mini
--evaluator-model venus-gemini-3.1-pro
```

`--checker-model`是旧版兼容参数；新配置统一使用`--evaluator-model`。

## 批量切换candidate

编辑`configs/experiments/baseline.yaml`：

```yaml
roles:
  candidate: venus-deepseek-v4-pro
  player: venus-deepseek-v4-pro
  evaluator: venus-deepseek-v4-pro

matrix:
  candidate:
    - venus-deepseek-v4-pro
    - venus-gpt-5.5
    - venus-gemini-3.1-pro
```

预检整个矩阵：

```bash
python3 -m runners.model_sweep \
  --experiment configs/experiments/baseline.yaml \
  --world assets/world_002 \
  --character yu_zecheng \
  --scenario assets/world_002/scenarios/yu_zecheng_archive_request.yaml \
  --dry-run
```

删除`--dry-run`后，每个组合使用独立目录和Checkpoint。单个模型失败会记录
在Sweep summary中，不会阻止后续模型。

## Player与Evaluator敏感性实验

参考`configs/experiments/sensitivity.example.yaml`。

只测试Player影响时，只展开Player：

```yaml
matrix:
  candidate:
    - venus-gpt-5.5
  player:
    - venus-deepseek-v4-pro
    - venus-gpt-5.4-mini
    - venus-gemini-3.8-flash
```

只测试Evaluator影响时，保持Player固定，展开Evaluator。全组合实验会计算
`candidate × player × evaluator`笛卡尔积，正式运行前必须用`--dry-run`
确认组合数。

## 调试流程

### 1. 配置和Runner预检

```bash
python3 -m runners.model_sweep \
  --experiment configs/experiments/baseline.yaml \
  --world assets/world_002 \
  --character yu_zecheng \
  --scenario assets/world_002/scenarios/yu_zecheng_archive_request.yaml \
  --dry-run
```

这一步检查模型名、角色分配、世界资产、调用上界和输出目录，不调用API。

### 2. 所有Venus模型基础连通测试

```bash
python3 -m runners.model_smoke \
  --registry configs/llm/registry.yaml
```

只测试一个模型：

```bash
python3 -m runners.model_smoke \
  --registry configs/llm/registry.yaml \
  --model venus-gpt-5.5
```

通过条件：

- HTTP调用成功；
- 输出可解析为JSON对象；
- `ok=true`；
- 未命中finish reason截断；
- completion token未耗尽配置预算。

### 3. 长上下文验证

```bash
python3 -m runners.model_smoke \
  --registry configs/llm/registry.yaml \
  --model venus-gpt-5.5 \
  --long-context assets/world_002/frozen/yu_zecheng/history.jsonl \
  --long-context-characters 120000
```

工具会在上下文首、中、尾放置不同marker，并要求模型原样返回。只证明API
确实保留三个位置，比“请求返回200”更能发现静默输入截断。

当前World 002实际Prompt约为：

- Stage 1 QA批次：107447–107669字符；
- Stage 2分支：约99685–99737字符。

因此12万字符是覆盖当前Stage 1-2的保守smoke规模。

### 4. 长输出压力测试

```bash
python3 -m runners.model_smoke \
  --registry configs/llm/registry.yaml \
  --model venus-deepseek-v4-pro \
  --output-characters 2000
```

这是严格指令跟随压力测试，不等价于正常Stage输出质量。失败可能来自：

- 模型没有严格生成指定字符数；
- 隐藏推理耗尽输出预算；
- `finish_reason=length/MAX_TOKENS`；
- 超时；
- JSON不完整。

正常实验是否可用，应再结合真实Stage Prompt smoke判断。

### 5. 小规模真实协议检查

保留完整Stage 1/2，同时把Stage 3缩短为一个seed和一个mode：

```bash
python3 -m runners.full_experiment \
  --world assets/world_002 \
  --character yu_zecheng \
  --scenario assets/world_002/scenarios/yu_zecheng_archive_request.yaml \
  --registry configs/llm/registry.yaml \
  --candidate-model venus-gpt-5.5 \
  --player-model venus-deepseek-v4-pro \
  --evaluator-model venus-deepseek-v4-pro \
  --seeds 11 \
  --modes normal \
  --turns 1
```

该命令仍会产生真实API费用。建议新模型先完成基础、长上下文测试，再执行。

## 重试、并发和截断

### 重试

会重试：

- 429；
- 网络连接错误；
- timeout；
- HTTP 408/409；
- HTTP 5xx；
- 临时缺失choices/content。

普通不可恢复4xx不会重试。重试次数是“总尝试次数”，不是额外重试次数。

### 并发

同一Python进程中，所有客户端按：

```text
backend base_url + remote_model
```

共享并发闸门。模型级`max_concurrency`优先于backend默认值；同一模型出现
多个限制时采用更小值。

该闸门不跨独立进程。如果同时启动多个`model_sweep`进程，多个进程的并发
会叠加；正式批量实验应只启动一个Sweep，或额外部署跨进程/集中式限流。

### 输出截断

以下情况直接抛出`APIClientError`：

- `finish_reason`为`length`、`MAX_TOKENS`或`max_output_tokens`；
- `completion_tokens >= GenerationConfig.max_tokens`；
- content为空；
- JSON或业务Schema经有限修复后仍不合法。

因此半截JSON不会进入GameCore或评分。

## 代码中直接调用注册模型

```python
from llm import load_model_registry

registry = load_model_registry(
    "configs/llm/registry.yaml",
    project_root=".",
)
client = registry.create_client(
    "venus-gpt-5.5",
    scene="manual_debug",
)
config = registry.generation_config("venus-gpt-5.5")

text = client.generate(
    system_prompt="You are a helpful assistant.",
    user_prompt='Return exactly {"ok":true}.',
    config=config,
)
```

业务代码应优先调用`generate_structured()`，不要自行解析JSON。

## 新增模型

### 已有平台、已有协议

例如新增一个Venus模型，只需在`configs/llm/registry.yaml`增加model：

```yaml
venus-new-model:
  backend: venus
  remote_model: new-model-id
  max_concurrency: 2
  timeout_seconds: 300
  max_attempts: 3
  capabilities:
    token_parameter: max_tokens
    temperature: true
    top_p: true
    seed: false
    structured_output: json_schema
  defaults:
    temperature: 0.0
    top_p: 1.0
    max_tokens: 4096
    seed: null
    max_format_retries: 2
```

然后依次执行：

1. 单模型基础smoke；
2. JSON Schema检查；
3. 12万字符marker检查；
4. 真实Stage小规模检查；
5. 加入baseline或sensitivity矩阵。

不要根据模型名称前缀猜参数，必须按真实接口测试结果填写capabilities。

### 新平台但兼容OpenAI Chat Completions

在`backends`中新增`type: openai_compatible`，再让model引用该backend。
通常不需要新增Python文件。

### 全新API协议

只有消息格式、认证或响应协议不兼容Chat Completions时，才需要：

1. 在`llm/`中新增backend adapter；
2. 在`factory.py`增加provider路由；
3. 将平台异常转换为`APIClientError`；
4. 接入统一usage、重试、并发和截断检测；
5. 补充单元测试和真实smoke。

## 删除模型

1. 先从`configs/experiments/*.yaml`的matrix删除；
2. 确认没有运行配置引用；
3. 再从`configs/llm/registry.yaml`删除model profile；
4. 不要删除共享backend，除非已经没有模型引用它。

注册表加载时会拒绝未知backend或未知模型名，因此不会静默切换到其他模型。

## 常见问题

### 提示“未注册模型”

检查命令使用的是稳定注册名，而不是远端model ID。例如使用
`venus-gpt-5.5`，不是`gpt-5.5`。

### 提示Venus Token未设置

确认`.env`中存在完整`VENUS_API_KEY`，或存在
`ENV_VENUS_OPENAPI_SECRET_ID`且backend配置了`api_key_suffix`。

### 短Prompt成功，Stage 1失败

依次检查：

1. 12万字符marker是否通过；
2. completion token是否耗尽；
3. 是否只返回部分QA；
4. 是否触发JSON格式修复；
5. timeout是否适合该模型。

部分QA不是整批失败，Runner会只补问缺失项。

### finish reason是stop，为什么仍判截断

部分代理部署在输出预算耗尽时仍返回`stop`。客户端还会比较
`completion_tokens`与配置预算，二者任一显示耗尽就按截断处理。

### 为什么不在Venus失败后自动改用DashScope

自动回退会改变实验变量，导致结果无法确认究竟来自哪个平台。平台切换必须
在注册表和实验矩阵中显式声明。
