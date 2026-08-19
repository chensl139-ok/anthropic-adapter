"""
Anthropic-to-OpenAI Proxy

A lightweight FastAPI service that translates the Anthropic Messages API
(/v1/messages) into OpenAI Chat Completions API (/v1/chat/completions),
so any OpenAI-compatible backend (vLLM, sglang, TGI, etc.) can be used
with Anthropic-protocol clients.

Designed to run as a sidecar container next to your inference server.
"""

import json
import os
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Anthropic-to-OpenAI Adapter", version="1.0.0")

# --- Configuration via environment ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "")
PORT = int(os.getenv("PORT", "8080"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(block: Any) -> str:
    """Extract text from an Anthropic content block."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict) and block.get("type") == "text":
        return block.get("text", "")
    return ""


def _anthropic_to_openai(body: dict) -> dict:
    """Convert Anthropic /v1/messages request body to OpenAI /v1/chat/completions."""
    messages: list[dict] = []

    # system field -> system message
    system = body.get("system")
    if system:
        if isinstance(system, list):
            system_text = " ".join(_extract_text(b) for b in system)
        else:
            system_text = str(system)
        if system_text.strip():
            messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            text = "".join(_extract_text(b) for b in content)
            messages.append({"role": role, "content": text})
        else:
            messages.append({"role": role, "content": str(content)})

    oai: dict = {
        "model": body.get("model") or DEFAULT_MODEL,
        "messages": messages,
        "max_tokens": body.get("max_tokens", 4096),
    }
    if "temperature" in body:
        oai["temperature"] = body["temperature"]
    if "top_p" in body:
        oai["top_p"] = body["top_p"]
    stop = body.get("stop_sequences")
    if stop:
        oai["stop"] = stop
    if body.get("stream"):
        oai["stream"] = True
    return oai


_STOP_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def _openai_to_anthropic(resp: dict, model: str) -> dict:
    """Convert OpenAI chat completion response to Anthropic format."""
    choice = resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    text = message.get("content", "") or ""
    finish = choice.get("finish_reason", "stop")
    usage = resp.get("usage", {})
    return {
        "id": resp.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": _STOP_MAP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# OpenAI passthrough (so both protocols work behind one port)
# ---------------------------------------------------------------------------

async def _passthrough(request: Request, path: str):
    headers = {"Content-Type": "application/json"}
    if BACKEND_API_KEY:
        headers["Authorization"] = f"Bearer {BACKEND_API_KEY}"
    body = await request.body()
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        resp = await client.request(
            request.method,
            f"{BACKEND_URL}{path}",
            content=body,
            headers=headers,
            params=request.query_params,
        )
    try:
        content = resp.json()
    except Exception:
        content = {"raw": resp.text}
    return JSONResponse(content=content, status_code=resp.status_code)


# ---------------------------------------------------------------------------
# Anthropic /v1/messages  (non-streaming + streaming)
# ---------------------------------------------------------------------------

@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    stream = body.get("stream", False)
    requested_model = body.get("model", DEFAULT_MODEL)
    oai_body = _anthropic_to_openai(body)

    headers = {"Content-Type": "application/json"}
    if BACKEND_API_KEY:
        headers["Authorization"] = f"Bearer {BACKEND_API_KEY}"

    if stream:
        return StreamingResponse(
            _stream_sse(oai_body, headers, requested_model),
            media_type="text/event-stream",
        )

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        resp = await client.post(
            f"{BACKEND_URL}/v1/chat/completions", json=oai_body, headers=headers
        )
    if resp.status_code != 200:
        return JSONResponse(
            status_code=resp.status_code,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": resp.text},
            },
        )
    return _openai_to_anthropic(resp.json(), requested_model)


async def _stream_sse(oai_body: dict, headers: dict, model: str):
    """Convert OpenAI streaming chunks to Anthropic SSE events."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # message_start
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    # content_block_start
    yield _sse("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })

    output_tokens = 0
    finish_reason = "stop"

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        async with client.stream(
            "POST", f"{BACKEND_URL}/v1/chat/completions",
            json=oai_body, headers=headers,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                usage = chunk.get("usage")
                if usage:
                    output_tokens = usage.get("completion_tokens", output_tokens)

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})
                text = delta.get("content", "")
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr
                if text:
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                    })

    # content_block_stop
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})

    # message_delta
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": _STOP_MAP.get(finish_reason, "end_turn"),
            "stop_sequence": None,
        },
        "usage": {"output_tokens": output_tokens},
    })

    # message_stop
    yield _sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# OpenAI passthrough routes + health
# ---------------------------------------------------------------------------

@app.api_route("/v1/chat/completions", methods=["GET", "POST"])
async def passthrough_chat(request: Request):
    return await _passthrough(request, "/v1/chat/completions")


@app.api_route("/v1/models", methods=["GET"])
async def passthrough_models(request: Request):
    return await _passthrough(request, "/v1/models")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
