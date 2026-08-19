# anthropic-adapter

[English](README.md) | [简体中文](README_CN.md)

A lightweight sidecar that translates the **Anthropic Messages API** (`/v1/messages`) into the **OpenAI Chat Completions API** (`/v1/chat/completions`), so any OpenAI-compatible inference backend (vLLM, sglang, TGI, etc.) can serve Anthropic-protocol clients — including tool calls, thinking/reasoning, and streaming.

## Features

**Anthropic protocol (`/v1/messages`)**

- Text generation, streaming and non-streaming
- **Tool calls**: `tools` with `input_schema`, `tool_choice`, `tool_result` blocks, assistant `tool_use` blocks
- **Thinking/reasoning**: maps `reasoning_content` from the backend to Anthropic `thinking` content blocks; supports `thinking.type` to toggle `enable_thinking` via `chat_template_kwargs`
- System prompt, `stop_sequences`, `temperature`, `top_p`, `top_k`
- Proper Anthropic SSE event sequence with block state machine (`thinking` -> `text` -> `tool_use`), including `thinking_delta`, `text_delta`, and `input_json_delta` events
- Correct `stop_reason` mapping (`end_turn`, `max_tokens`, `tool_use`)

**OpenAI passthrough (`/v1/chat/completions`, `/v1/models`)**

- Streaming and non-streaming passthrough to backend, so OpenAI-protocol clients still work behind the same port

## Quick Start

### Docker

```bash
docker run -d \
  -p 8080:8080 \
  -e BACKEND_URL=http://127.0.0.1:8000 \
  -e BACKEND_API_KEY=your-key \
  -e DEFAULT_MODEL=your-model-name \
  ghcr.io/chensl139-ok/anthropic-adapter:2.0.0
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://127.0.0.1:8000` | OpenAI-compatible backend URL |
| `BACKEND_API_KEY` | _(empty)_ | API key forwarded as `Authorization: Bearer` |
| `DEFAULT_MODEL` | _(empty)_ | Fallback model name if request doesn't specify one |
| `PORT` | `8080` | Listen port |

### Examples

```bash
# Text generation (non-streaming)
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: any" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "your-model",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: any" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "your-model",
    "max_tokens": 128,
    "stream": true,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Tool calls
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: any" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "your-model",
    "max_tokens": 256,
    "tools": [{
      "name": "get_weather",
      "description": "Get weather for a city",
      "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }],
    "messages": [{"role": "user", "content": "What is the weather in Beijing?"}]
  }'

# Disable thinking
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: any" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "your-model",
    "max_tokens": 128,
    "thinking": {"type": "disabled"},
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Sidecar Example (GPU Function / K8s)

Add as a `sidecarContainers` entry in your pod spec:

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

Then point your gateway entrypointPort to `8080`.

## Protocol Mapping

| Anthropic | OpenAI |
|---|---|
| `system` field | `role: system` message |
| `content` blocks (text) | `content` string |
| `tools[].input_schema` | `tools[].function.parameters` |
| `tool_choice: auto / any / tool` | `auto / required / {type: function, function: {name}}` |
| `tool_result` content blocks | `role: tool` messages with `tool_call_id` |
| assistant `tool_use` blocks | `tool_calls` array |
| `thinking.type: disabled` | `chat_template_kwargs: {enable_thinking: false}` |
| `thinking.type: enabled` | `chat_template_kwargs: {enable_thinking: true}` |
| `stop_sequences` | `stop` |
| `top_k` | `top_k` |
| `reasoning_content` (response) | `thinking` content block |
| `tool_calls` (response) | `tool_use` content blocks |
| `finish_reason: stop / length / tool_calls` | `stop_reason: end_turn / max_tokens / tool_use` |

## License

MIT
