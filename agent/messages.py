"""对话历史管理（conversation 核心）。

三件套：
  - Message(role, content, tool_calls, tool_call_id)：发给模型的一条消息（支持原生 tool calling）。
  - Turn(action, args, result)：结构化轨迹，供 context 压缩。
  - Conversation：同时持有 messages（发给模型的序列）与 turns（结构化轨迹），
    是活上下文，跨任务累积，不落盘、不恢复。

契约（DESIGN.md §3）：
  - set_system / add_task / append_assistant_tool_calls / append_tool_result / add_final
  - as_openai() -> [{role, content?, tool_calls?, tool_call_id?}]
  - total_chars() -> int
"""

import json
from dataclasses import dataclass

from agent.tools import ToolResult


@dataclass(frozen=True)
class Message:
    """一条消息。role 允许 system / user / assistant / tool。"""

    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None


@dataclass(frozen=True)
class Turn:
    """一轮工具调用：哪个工具 + 参数 + 结果。"""

    action: str
    args: dict
    result: ToolResult


def format_observation(tool: str, result: ToolResult) -> str:
    """把 ToolResult 转成工具结果消息的正文（对齐 DESIGN.md §6）。"""
    if result.ok:
        body = {"tool": tool, "ok": True, "output": result.output}
    else:
        body = {"tool": tool, "ok": False, "error_type": result.error, "message": result.output}
    return json.dumps(body, ensure_ascii=False)


class Conversation:
    """活上下文：messages（发给模型）+ turns（结构化，供压缩）。"""

    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.turns: list[Turn] = []
        self.last_task: str = ""

    def set_system(self, rules: str) -> None:
        """设置（或替换）系统提示词，始终作为第一条消息。"""
        self.messages = [m for m in self.messages if m.role != "system"]
        self.messages.insert(0, Message("system", rules))

    def add_task(self, text: str) -> None:
        """追加一条新任务（user 消息），并记录为当前任务。"""
        self.messages.append(Message("user", text))
        self.last_task = text

    def add_final(self, content: str) -> None:
        """追加 final 回答（assistant 消息），让下个任务看到上一轮总结。"""
        self.messages.append(Message("assistant", content))

    def append_assistant_tool_calls(self, response) -> None:
        """追加 assistant 的 tool_calls 消息。tool_calls 是含 id/name/arguments 的对象列表。"""
        tool_calls = response.tool_calls
        openai_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.raw_arguments if getattr(tc, "raw_arguments", "{}") != "{}"
                    else json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in tool_calls
        ]
        self.messages.append(
            Message("assistant", tool_calls=openai_calls, reasoning_content=response.reasoning_content)
        )

    def append_tool_result(self, tool_call_id: str, name: str, args: dict, result: ToolResult) -> None:
        """追加 tool 结果消息 + 一条结构化 Turn。"""
        self.messages.append(
            Message("tool", content=format_observation(name, result), tool_call_id=tool_call_id, name=name)
        )
        self.turns.append(Turn(name, args, result))

    def as_openai(self) -> list[dict]:
        """转成 chat 接口需要的消息列表。"""
        out = []
        for m in self.messages:
            d = {"role": m.role}
            if m.content is not None:
                d["content"] = m.content
            if m.tool_calls is not None:
                if m.role == "assistant" and "content" not in d:
                    d["content"] = None
                d["tool_calls"] = m.tool_calls
            if m.tool_call_id is not None:
                d["tool_call_id"] = m.tool_call_id
            if m.name is not None:
                d["name"] = m.name
            if m.reasoning_content is not None:
                d["reasoning_content"] = m.reasoning_content
            out.append(d)
        return out

    def total_chars(self) -> int:
        """字符预算：所有消息内容长度之和，供 context 判断是否超阈值。"""
        return sum(len(m.content or "") for m in self.messages)
