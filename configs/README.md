# Configs

该目录保存：

- 模型ID、版本、采样参数和上下文限制；
- Normal与Pressure Player配置；
- pilot、主实验和消融实验矩阵；
- 每个场景允许加载的4–6个Action名称。

配置不得包含私有测试答案。

`models/venus_deepseek.yaml`是默认配置，通过Venus OpenAPI调用内部
`deepseek-v4-pro`；真实Token只写入项目根目录`.env`的
`VENUS_API_KEY`。`models/qwen_pilot.yaml`仅保留为显式旧后端配置，
不会被默认入口选择。

`llm/registry.yaml`是跨平台模型注册表，分别声明backend连接信息和model
能力。`experiments/*.yaml`只引用注册表中的稳定模型名，并通过
`candidate`、`player`、`evaluator`角色或`matrix`定义主实验与敏感性
实验。Runner不应包含远端模型ID或供应商特判。
