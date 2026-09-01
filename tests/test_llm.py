"""llm.py 的单元测试。

覆盖 DESIGN.md §3 llm.py：
  - complete：构造请求（model/messages/temperature/Authorization/endpoint）并取回 content
  - summarize：作为语义 summarizer，用摘要 prompt 调用模型
  - 错误：缺 key / 缺 model / HTTP 失败 / 响应缺字段
  - build_summarize_messages / _render_turn 的渲染与截断
"""

import json
import socket
import urllib.error

import pytest

from agent.config import Config
from agent.messages import Turn
from agent.llm import (
    LLM,
    LLMError,
    _parse_tool_call,
    _render_turn,
    build_summarize_messages,
)
from agent.tools import ToolResult


def turn(action, args, ok=True, error="", output=""):
    return Turn(action, args, ToolResult(ok=ok, output=output, error=error))


def config(**kw):
    base = {"api_key": "sk", "model": "gpt-4o", "base_url": "http://x/v1", "temperature": 0.0}
    base.update(kw)
    return Config(**base)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


class FakeHTTPError(urllib.error.HTTPError):
    def __init__(self, payload):
        super().__init__("http://x/v1/chat/completions", 400, "Bad Request", hdrs=None, fp=None)
        self._payload = payload

    def read(self):
        return self._payload.encode("utf-8")


def make_urlopen(content, captured=None):
    def fake(req, timeout=None):
        if captured is not None:
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["auth"] = req.get_header("Authorization")
            captured["timeout"] = timeout
        return FakeResponse({"choices": [{"message": {"content": content}}]})

    return fake


# ---- complete ----

def test_complete_returns_content():
    llm = LLM(config(), urlopen=make_urlopen('{"action": "final"}'))
    resp = llm.complete([{"role": "user", "content": "hi"}])
    assert resp.content == '{"action": "final"}'
    assert resp.tool_calls == []


def test_complete_parses_tool_calls():
    payload = {
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "list_files", "arguments": '{"path": "."}'}}
        ]}}]
    }
    llm = LLM(config(), urlopen=lambda req, timeout=None: FakeResponse(payload))
    resp = llm.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "list_files"}}])
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "list_files"
    assert resp.tool_calls[0].arguments == {"path": "."}


def test_complete_preserves_reasoning_content():
    payload = {
        "choices": [{"message": {
            "content": None,
            "reasoning_content": "hidden chain",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "list_files", "arguments": '{"path": "."}'}}
            ],
        }}]
    }
    llm = LLM(config(), urlopen=lambda req, timeout=None: FakeResponse(payload))
    resp = llm.complete([{"role": "user", "content": "hi"}])
    assert resp.reasoning_content == "hidden chain"


def test_parse_tool_call_keeps_argument_parse_error():
    tc = _parse_tool_call({
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":'},
    })
    assert tc.arguments == {}
    assert tc.raw_arguments == '{"path":'
    assert "不是合法 JSON" in tc.parse_error


def test_complete_sends_tools():
    captured = {}
    llm = LLM(config(), urlopen=make_urlopen("ok", captured))
    tools = [{"type": "function", "function": {"name": "list_files", "parameters": {}}}]
    llm.complete([{"role": "user", "content": "hi"}], tools=tools)
    assert captured["body"]["tools"] == tools
    assert captured["body"]["tool_choice"] == "auto"


def test_complete_builds_request():
    captured = {}
    llm = LLM(config(), urlopen=make_urlopen("ok", captured))
    llm.complete([{"role": "user", "content": "hi"}])
    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["body"] == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
    }
    assert captured["auth"] == "Bearer sk"


# ---- summarize ----

def test_summarize_empty_content_raises():
    llm = LLM(config(), urlopen=make_urlopen("   "))
    with pytest.raises(LLMError):
        llm.summarize([turn("list_files", {"path": "."}, ok=True)])


def test_summarize_uses_summary_prompt():
    captured = {}
    llm = LLM(config(), urlopen=make_urlopen("摘要", captured))
    turns = [turn("read_file", {"path": "a.py"}, ok=True, output="x=1")]
    assert llm.summarize(turns) == "摘要"
    msgs = captured["body"]["messages"]
    assert msgs[0]["role"] == "system"
    assert "已完成" in msgs[0]["content"]
    assert "read_file" in msgs[1]["content"]
    assert "a.py" in msgs[1]["content"]


# ---- 配置与错误 ----

def test_missing_key_raises():
    with pytest.raises(LLMError):
        LLM(Config(api_key="", model="m"))


def test_missing_model_raises():
    with pytest.raises(LLMError):
        LLM(Config(api_key="k", model=""))


def test_endpoint_default_base():
    llm = LLM(Config(api_key="k", model="m", base_url=""))
    assert llm._endpoint() == "https://api.openai.com/v1/chat/completions"


def test_http_error_raises():
    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    llm = LLM(config(), urlopen=boom)
    with pytest.raises(LLMError):
        llm.complete([{"role": "user", "content": "hi"}])


def test_http_error_includes_response_body():
    def boom(req, timeout=None):
        raise FakeHTTPError('{"error":{"message":"messages with role tool must include name"}}')

    llm = LLM(config(), urlopen=boom)
    with pytest.raises(LLMError) as exc:
        llm.complete([{"role": "user", "content": "hi"}])
    assert "HTTP Error 400" in str(exc.value)
    assert "messages with role tool" in str(exc.value)


def test_timeout_error_raises_llmerror():
    def boom(req, timeout=None):
        raise TimeoutError("timed out")

    llm = LLM(config(), urlopen=boom)
    with pytest.raises(LLMError) as exc:
        llm.complete([{"role": "user", "content": "hi"}])
    assert "超时" in str(exc.value)


def test_socket_timeout_error_raises_llmerror():
    def boom(req, timeout=None):
        raise socket.timeout("timed out")

    llm = LLM(config(), urlopen=boom)
    with pytest.raises(LLMError) as exc:
        llm.complete([{"role": "user", "content": "hi"}])
    assert "超时" in str(exc.value)


def test_malformed_response_raises():
    def fake(req, timeout=None):
        return FakeResponse({"unexpected": True})

    llm = LLM(config(), urlopen=fake)
    with pytest.raises(LLMError):
        llm.complete([{"role": "user", "content": "hi"}])


# ---- 渲染与截断 ----

def test_render_turn_failure():
    line = _render_turn(
        turn(
            "run_command",
            {"command": "python -m pytest"},
            ok=False,
            error="CommandFailed",
            output="AssertionError: expected 1",
        )
    )
    assert "run_command" in line
    assert "失败(CommandFailed)" in line
    assert "AssertionError" in line


def test_render_turn_truncates_args():
    line = _render_turn(turn("write_file", {"path": "a.py", "content": "x" * 100}, ok=True, output="created"))
    assert "a.py" in line
    assert "x" * 100 not in line


def test_build_summarize_messages_truncates_output():
    turns = [turn("read_file", {"path": "a.py"}, ok=True, output="x" * 1000)]
    msgs = build_summarize_messages(turns)
    assert "x" * 1000 not in msgs[1]["content"]
    assert "x" * 500 in msgs[1]["content"]


def test_render_turn_truncates_failure_output():
    line = _render_turn(
        turn(
            "run_command",
            {"command": "python -m pytest"},
            ok=False,
            error="CommandFailed",
            output="x" * 1000,
        )
    )
    assert "x" * 1000 not in line
    assert "x" * 500 in line
