# anthropic-adapter

A lightweight sidecar that translates the **Anthropic Messages API** (`/v1/messages`) into the **OpenAI Chat Completions API** (`/v1/chat/completions`), so any OpenAI-compatible inference backend (vLLM, sglang, TGI, etc.) can serve Anthropic-protocol clients.

## Features

- `/v1/messages` (Anthropic) with **streaming** and **non-streaming** support
- `/v1/chat/completions` and `/v1/models` **passthrough** to backend (OpenAI protocol still works)
- System prompt, `stop_sequences`, `temperature`, `top_p` mapping
- Proper Anthropic SSE event sequence (`message_start` -> `content_block_start` -> `content_block_delta` -> `content_block_stop` -> `message_delta` -> `message_stop`)
- Single Python file, ~280 lines, minimal dependencies

## Quick Start

### Docker

```bash
docker run -d \
  -p 8080:8080 \
  -e BACKEND_URL=http://127.0.0.1:8000 \
  -e BACKEND_API_KEY=your-key \
  -e DEFAULT_MODEL=your-model-name \
  ghcr.io/chensl139-ok/anthropic-adapter:latest
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://127.0.0.1:8000` | OpenAI-compatible backend URL |
| `BACKEND_API_KEY` | _(empty)_ | API key forwarded as `Authorization: Bearer` |
| `DEFAULT_MODEL` | _(empty)_ | Fallback model name if request doesn't specify one |
| `PORT` | `8080` | Listen port |

### Test

```bash
# Anthropic protocol (non-streaming)
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
```

### Sidecar Example (GPU Function / K8s)

Add as a `sidecarContainers` entry in your pod spec:

```yaml
sidecarContainers:
  - name: anthropic-adapter
    image: ghcr.io/chensl139-ok/anthropic-adapter:latest
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

## License

MIT
