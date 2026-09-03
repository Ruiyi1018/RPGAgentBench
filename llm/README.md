# LLM

该目录独立维护模型调用层，不包含NPC、Player或Checker行为。

- `base.py`：供应商无关接口、生成参数、JSON解析和格式重试；
- `venus.py`：Venus OpenAPI异步适配器及现有同步调用链的兼容边界；
- `factory.py`：严格按provider创建客户端，未知provider不会回退；
- `openai_compatible.py`：保留的显式DashScope兼容客户端；
- `config.py`：读取`configs/models/*.yaml`并自动加载项目根目录`.env`；
- `__init__.py`：对外导出稳定接口。

密钥只写入项目根目录`.env`，模型ID、端点与采样参数写入`configs/models/`。
默认配置为`configs/models/venus_deepseek.yaml`，固定调用内部部署的
`deepseek-v4-pro`。
