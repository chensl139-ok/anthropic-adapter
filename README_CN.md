# anthropic-adapter

[English](README.md) | 简体中文

一个轻量级 sidecar 代理，将 **Anthropic Messages API**（`/v1/messages`）转换为 **OpenAI Chat Completions API**（`/v1/chat/completions`），使任何 OpenAI 兼容的推理后端（vLLM、sglang、TGI 等）都能支持 Anthropic 协议客户端 —— 包括工具调用、思考/推理、以及流式输出。

## 功能

**Anthropic 协议（`/v1/messages`）**

- 文本生成，支持流式和非流式
- **工具调用**：`tools` + `input_schema`、`tool_choice`、`tool_result` 回传块、assistant `tool_use` 块
- **思考/推理**：将后端的 `reasoning_content` 映射为 Anthropic `thinking` 内容块；支持通过 `thinking.type` 配合 `chat_template_kwargs` 控制 `enable_thinking` 开关
- System prompt、`stop_sequences`、`temperature`、`top_p`、`top_k`
- 完整的 Anthropic SSE 事件序列，块状态机自动切换（`thinking` -> `text` -> `tool_use`），包含 `thinking_delta`、`text_delta`、`input_json_delta` 事件
- 正确的 `stop_reason` 映射（`end_turn`、`max_tokens`、`tool_use`）

**OpenAI 透传（`/v1/chat/completions`、`/v1/models`）**

- 流式和非流式直接透传后端，OpenAI 协议客户端在同一端口照常使用

## 快速开始

### Docker

```bash
docker run -d \
  -p 8080:8080 \
  -e BACKEND_URL=http://127.0.0.1:8000 \
  -e BACKEND_API_KEY=your-key \
  -e DEFAULT_MODEL=your-model-name \
  ghcr.io/chensl139-ok/anthropic-adapter:2.0.0
```

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `BACKEND_URL` | `http://127.0.0.1:8000` | OpenAI 兼容后端地址 |
| `BACKEND_API_KEY` | _(空)_ | 转发为 `Authorization: Bearer` 的 API Key |
| `DEFAULT_MODEL` | _(空)_ | 请求未指定模型时的默认模型名 |
| `PORT` | `8080` | 监听端口 |

### 示例

```bash
# 文本生成（非流式）
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: any" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "your-model",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "你好！"}]
  }'

# 流式输出
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: any" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "your-model",
    "max_tokens": 128,
    "stream": true,
    "messages": [{"role": "user", "content": "你好！"}]
  }'

# 工具调用
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: any" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "your-model",
    "max_tokens": 256,
    "tools": [{
      "name": "get_weather",
      "description": "获取城市天气",
      "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }],
    "messages": [{"role": "user", "content": "北京天气怎么样？"}]
  }'

# 关闭思考
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: any" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "your-model",
    "max_tokens": 128,
    "thinking": {"type": "disabled"},
    "messages": [{"role": "user", "content": "你好！"}]
  }'
```

### Sidecar 示例（GPU 云函数 / K8s）

在 Pod 配置中添加 `sidecarContainers`：

```yaml
sidecarContainers:
  - name: anthropic-adapter
    image: ghcr.io/chensl139-ok/anthropic-adapter:2.0.0
    ports:
      - containerPort: 8080
        portName: anthropic
    env:
      - name: BACKEND_URL
        value: 'http://127.0.0.1:8000'
      - name: DEFAULT_MODEL
        value: 'your-model-name'
    readinessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 10
```

然后将网关入口端口指向 `8080`。

## 协议映射

| Anthropic | OpenAI |
|---|---|
| `system` 字段 | `role: system` 消息 |
| `content` 块（text） | `content` 字符串 |
| `tools[].input_schema` | `tools[].function.parameters` |
| `tool_choice: auto / any / tool` | `auto / required / {type: function, function: {name}}` |
| `tool_result` 内容块 | `role: tool` 消息 + `tool_call_id` |
| assistant `tool_use` 块 | `tool_calls` 数组 |
| `thinking.type: disabled` | `chat_template_kwargs: {enable_thinking: false}` |
| `thinking.type: enabled` | `chat_template_kwargs: {enable_thinking: true}` |
| `stop_sequences` | `stop` |
| `top_k` | `top_k` |
| `reasoning_content`（响应） | `thinking` 内容块 |
| `tool_calls`（响应） | `tool_use` 内容块 |
| `finish_reason: stop / length / tool_calls` | `stop_reason: end_turn / max_tokens / tool_use` |

## License

MIT
