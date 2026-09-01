"""模型调用封装：OpenAI 兼容 /chat/completions，只用标准库 urllib。

两个接口（对应 DESIGN.md §3）：
  - complete(messages, tools) -> ModelResponse  主循环：消息列表 + 工具 schema -> 结构化响应
  - summarize(turns) -> str                     语义压缩：旧轨迹 -> 「已完成/仍需」摘要

合规：不用 agent 框架/SDK、托管执行、文件工具；HTTP 用标准库，API 契约显式可见。
"""

import json
import socket
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.config import Config
from agent.messages import Turn

DEFAULT_BASE_URL = "https://api.openai.com/v1"
CHAT_PATH = "/chat/completions"
REQUEST_TIMEOUT = 60

SUMMARIZE_SYSTEM = (
    "你是摘要器。把下面 coding agent 的执行轨迹压缩成简洁摘要，格式：\n"
    "已完成：\n- ...\n仍需：\n- ...\n"
    "必须保留未完成事项（失败的工具调用）与关键结论（改动了哪些文件、测试结果）。"
)
_ARGS_MAX = 40
_OUTPUT_MAX = 500


class LLMError(Exception):
    """模型调用失败（缺配置 / 网络 / HTTP / 响应格式）。"""


@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = "{}"
    parse_error: str = ""


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    reasoning_content: str | None = None


class LLM:
    """OpenAI 兼容模型的薄封装。urlopen 可注入以便测试。"""

    def __init__(self, config: Config, urlopen: Callable[..., Any] | None = None):
        if not config.api_key:
            raise LLMError("缺少 OPENAI_API_KEY，请设置环境变量或 .env")
        if not config.model:
            raise LLMError("缺少 CODING_AGENT_MODEL，请设置环境变量或 .env")
        self.config = config
        self._urlopen = urlopen or urllib.request.urlopen

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> ModelResponse:
        """主循环：消息列表 + 工具 schema -> 结构化响应（content 或 tool_calls）。"""
        return self._chat(messages, tools=tools)

    def summarize(self, turns: list[Turn]) -> str:
        """语义压缩：旧轨迹 -> 摘要（独立非 tool-calling 路径）。"""
        summary = self._chat_text(build_summarize_messages(turns))
        if not summary.strip():
            raise LLMError("summarize 返回空内容")
        return summary

    # ---- 内部 ----

    def _chat(self, messages: list[dict], tools: list[dict] | None = None) -> ModelResponse:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        data = self._post(payload)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"响应缺少 choices[0].message: {data!r}") from exc
        return ModelResponse(
            content=message.get("content"),
            tool_calls=[_parse_tool_call(tc) for tc in message.get("tool_calls", [])],
            reasoning_content=message.get("reasoning_content"),
        )

    def _chat_text(self, messages: list[dict]) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        data = self._post(payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"响应缺少 choices[0].message.content: {data!r}") from exc

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint(),
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with self._urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = _read_error_body(exc)
            detail = f"HTTP Error {exc.code}: {exc.reason}"
            if body:
                detail += f" | {body}"
            raise LLMError(f"模型调用失败: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"模型调用失败: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise LLMError(f"模型调用超时: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"响应不是合法 JSON: {exc}") from exc

    def _endpoint(self) -> str:
        base = (self.config.base_url or DEFAULT_BASE_URL).rstrip("/")
        return base + CHAT_PATH


def _parse_tool_call(tc: dict) -> ModelToolCall:
    fn = tc.get("function", {})
    raw = fn.get("arguments") or "{}"
    parse_error = ""
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError as exc:
        arguments = {}
        parse_error = f"arguments 不是合法 JSON: {exc}"
    if not isinstance(arguments, dict):
        arguments = {}
        parse_error = "arguments 必须是 JSON object"
    return ModelToolCall(
        id=tc.get("id", ""),
        name=fn.get("name", ""),
        arguments=arguments,
        raw_arguments=raw,
        parse_error=parse_error,
    )


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
    return _clip(body, 1000)


def build_summarize_messages(turns: list[Turn]) -> list[dict]:
    """把旧轨迹渲染成摘要请求的消息（纯函数，可独立测试）。"""
    lines = [_render_turn(t) for t in turns]
    user = "以下是一段 agent 执行轨迹：\n" + "\n".join(lines) + "\n请给出摘要。"
    return [
        {"role": "system", "content": SUMMARIZE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _render_turn(turn: Turn) -> str:
    args = ", ".join(f"{k}={_clip(str(v), _ARGS_MAX)}" for k, v in turn.args.items())
    label = f"{turn.action}({args})" if args else turn.action
    if not turn.result.ok:
        output = _clip(turn.result.output, _OUTPUT_MAX)
        return f"- {label}: 失败({turn.result.error})\n  输出: {output}"
    output = _clip(turn.result.output, _OUTPUT_MAX)
    return f"- {label}: 成功\n  输出: {output}"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
