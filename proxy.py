"""
Anthropic-to-OpenAI Proxy

A lightweight FastAPI service that translates the Anthropic Messages API
(/v1/messages) into OpenAI Chat Completions API (/v1/chat/completions),
so any OpenAI-compatible backend (vLLM, sglang, TGI, etc.) can be used
with Anthropic-protocol clients.

Supports: text, thinking (reasoning), tool calls, streaming, and OpenAI passthrough.
"""

import json
import os
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Anthropic-to-OpenAI Adapter", version="2.0.0")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "")
PORT = int(os.getenv("PORT", "8080"))

_STOP_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def _extract_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, dict) and block.get("type") == "text":
        return block.get("text", "")
    return ""


# ---------------------------------------------------------------------------
# Request: Anthropic -> OpenAI
# ---------------------------------------------------------------------------

def _convert_tools(anthropic_tools: list) -> list:
    """Convert Anthropic tools to OpenAI function-calling format."""
    oai_tools = []
    for tool in anthropic_tools:
        if tool.get("type") == "computer" or tool.get("type") == "bash" or tool.get("type") == "text_editor":
            continue  # unsupported tool types
        oai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return oai_tools


def _convert_tool_choice(tc: Any) -> Any:
    """Convert Anthropic tool_choice to OpenAI tool_choice."""
    if isinstance(tc, str):
        return "auto"
    if isinstance(tc, dict):
        t = tc.get("type")
        if t == "auto":
            return "auto"
        if t == "any":
            return "required"
        if t == "tool":
            return {"type": "function", "function": {"name": tc.get("name", "")}}
    return "auto"


def _convert_messages(body: dict) -> list:
    """Convert Anthropic messages (including tool_result, tool_use) to OpenAI format."""
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
            continue

        if not isinstance(content, list):
            messages.append({"role": role, "content": str(content)})
            continue

        # Check if this is a tool_result message (user role with tool_result blocks)
        has_tool_result = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
        if has_tool_result:
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        tool_content = " ".join(_extract_text(b) for b in tool_content)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": str(tool_content) if tool_content else "",
                    })
                    # Any non-tool_result blocks in the same message become a follow-up user message
            # Pick up any text blocks that accompany the tool_result
            extra_text = "".join(_extract_text(b) for b in content if isinstance(b, dict) and b.get("type") == "text")
            if extra_text.strip():
                messages.append({"role": "user", "content": extra_text})
            continue

        # Assistant message with tool_use blocks -> reproduce as assistant with tool_calls
        has_tool_use = any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
        if has_tool_use and role == "assistant":
            text_parts = []
            tool_calls = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
            assistant_msg: dict = {"role": "assistant"}
            if text_parts:
                assistant_msg["content"] = "".join(text_parts)
            else:
                assistant_msg["content"] = None
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
            continue

        # Normal content blocks -> concatenate text
        text = "".join(_extract_text(b) for b in content)
        messages.append({"role": role, "content": text if text else ""})

    return messages


def _anthropic_to_openai(body: dict) -> dict:
    """Convert Anthropic /v1/messages request to OpenAI /v1/chat/completions."""
    messages = _convert_messages(body)
    oai: dict = {
        "model": body.get("model") or DEFAULT_MODEL,
        "messages": messages,
        "max_tokens": body.get("max_tokens", 4096),
    }

    # Sampling parameters
    if "temperature" in body:
        oai["temperature"] = body["temperature"]
    if "top_p" in body:
        oai["top_p"] = body["top_p"]
    if "top_k" in body:
        oai["top_k"] = body["top_k"]
    stop = body.get("stop_sequences")
    if stop:
        oai["stop"] = stop

    # Tools
    tools = body.get("tools")
    if tools:
        oai["tools"] = _convert_tools(tools)
    tc = body.get("tool_choice")
    if tc:
        oai["tool_choice"] = _convert_tool_choice(tc)

    # Thinking / reasoning control
    thinking = body.get("thinking")
    if thinking and isinstance(thinking, dict):
        if thinking.get("type") == "disabled":
            # Disable thinking via chat template kwargs (sglang/vLLM Qwen3 convention)
            oai["chat_template_kwargs"] = {"enable_thinking": False}
        elif thinking.get("type") == "enabled":
            # Ensure thinking is on (it's on by default for Qwen3)
            oai["chat_template_kwargs"] = {"enable_thinking": True}

    if body.get("stream"):
        oai["stream"] = True
    return oai


# ---------------------------------------------------------------------------
# Response: OpenAI -> Anthropic (non-streaming)
# ---------------------------------------------------------------------------

def _openai_to_anthropic(resp: dict, model: str) -> dict:
    """Convert OpenAI chat completion response to Anthropic format."""
    choice = resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    finish = choice.get("finish_reason", "stop")
    usage = resp.get("usage", {})

    content_blocks: list[dict] = []

    # Thinking block (reasoning_content)
    reasoning = message.get("reasoning_content")
    if reasoning:
        content_blocks.append({"type": "thinking", "thinking": reasoning, "signature": "signature_placeholder"})

    # Text block
    text = message.get("content") or ""
    if text:
        content_blocks.append({"type": "text", "text": text})

    # Tool use blocks
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        try:
            tool_input = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, KeyError):
            tool_input = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": tc.get("function", {}).get("name", ""),
            "input": tool_input,
        })

    return {
        "id": resp.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
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
# Streaming: OpenAI SSE -> Anthropic SSE
# ---------------------------------------------------------------------------

async def _stream_sse(oai_body: dict, headers: dict, model: str):
    """Convert OpenAI streaming chunks to Anthropic SSE events.

    Block state machine: idle -> thinking -> text -> tool_use(s) -> done
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    block_index = 0
    current_block_type = None  # None, "thinking", "text", "tool_use"
    output_tokens = 0
    input_tokens = 0
    finish_reason = "stop"

    # Track tool_use blocks: {tc_index: {block_index, id, name, args_buffer}}
    tool_blocks: dict[int, dict] = {}

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

                # Track usage
                usage = chunk.get("usage")
                if usage:
                    output_tokens = usage.get("completion_tokens", output_tokens)
                    input_tokens = usage.get("prompt_tokens", input_tokens)

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr

                # --- Reasoning content -> thinking block ---
                rc = delta.get("reasoning_content")
                if rc:
                    if current_block_type != "thinking":
                        if current_block_type is not None:
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                            block_index += 1
                        current_block_type = "thinking"
                        yield _sse("content_block_start", {
                            "type": "content_block_start", "index": block_index,
                            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                        })
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta", "index": block_index,
                        "delta": {"type": "thinking_delta", "thinking": rc},
                    })

                # --- Text content -> text block ---
                text = delta.get("content")
                if text:
                    if current_block_type != "text":
                        if current_block_type is not None:
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                            block_index += 1
                        current_block_type = "text"
                        yield _sse("content_block_start", {
                            "type": "content_block_start", "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta", "index": block_index,
                        "delta": {"type": "text_delta", "text": text},
                    })

                # --- Tool calls -> tool_use blocks ---
                tcs = delta.get("tool_calls")
                if tcs:
                    for tc in tcs:
                        tc_idx = tc.get("index", 0)
                        if tc_idx not in tool_blocks:
                            # Close previous block if any
                            if current_block_type is not None:
                                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                                block_index += 1
                            current_block_type = "tool_use"
                            tool_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"
                            tool_name = (tc.get("function") or {}).get("name") or ""
                            tool_blocks[tc_idx] = {
                                "block_index": block_index,
                                "id": tool_id,
                                "name": tool_name,
                                "args_buffer": "",
                            }
                            yield _sse("content_block_start", {
                                "type": "content_block_start", "index": block_index,
                                "content_block": {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}},
                            })
                        else:
                            # Update name if it was empty (streamed in later chunks)
                            fn_name = (tc.get("function") or {}).get("name")
                            if fn_name and not tool_blocks[tc_idx]["name"]:
                                tool_blocks[tc_idx]["name"] = fn_name

                        # Stream argument fragments
                        args_delta = (tc.get("function") or {}).get("arguments")
                        if args_delta:
                            tool_blocks[tc_idx]["args_buffer"] += args_delta
                            yield _sse("content_block_delta", {
                                "type": "content_block_delta", "index": tool_blocks[tc_idx]["block_index"],
                                "delta": {"type": "input_json_delta", "partial_json": args_delta},
                            })

    # Close the last open block
    if current_block_type is not None:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": _STOP_MAP.get(finish_reason, "end_turn"),
            "stop_sequence": None,
        },
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# OpenAI passthrough
# ---------------------------------------------------------------------------

async def _passthrough(request: Request, path: str):
    headers = {"Content-Type": "application/json"}
    if BACKEND_API_KEY:
        headers["Authorization"] = f"Bearer {BACKEND_API_KEY}"
    body = await request.body()

    is_stream = False
    if request.method == "POST" and body:
        try:
            parsed = json.loads(body)
            is_stream = parsed.get("stream", False)
        except Exception:
            pass

    if is_stream:
        async def _stream():
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                async with client.stream(
                    request.method, f"{BACKEND_URL}{path}",
                    content=body, headers=headers, params=request.query_params,
                ) as resp:
                    async for chunk in resp.aiter_raw():
                        yield chunk
        return StreamingResponse(_stream(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        resp = await client.request(
            request.method, f"{BACKEND_URL}{path}",
            content=body, headers=headers, params=request.query_params,
        )
    try:
        content = resp.json()
    except Exception:
        content = {"raw": resp.text}
    return JSONResponse(content=content, status_code=resp.status_code)


# ---------------------------------------------------------------------------
# Routes
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
            content={"type": "error", "error": {"type": "api_error", "message": resp.text}},
        )
    return _openai_to_anthropic(resp.json(), requested_model)


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
