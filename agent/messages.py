"""对话历史管理（conversation 核心）。

三件套：
  - Message(role, content)：发给模型的一条消息。
  - Turn(action, args, result)：结构化轨迹，供 context 压缩。
  - Conversation：同时持有 messages（发给模型的序列）与 turns（结构化轨迹），
    是活上下文，跨任务累积，不落盘、不恢复。

契约（DESIGN.md §3）：
  - set_system / add_task / append_turn
  - as_openai() -> [{role, content}]
  - total_chars() -> int
"""

import json
from dataclasses import dataclass

from agent.tools import ToolResult


@dataclass(frozen=True)
class Message:
    """一条消息。role 只允许 system / user / assistant。"""

    role: str
    content: str


@dataclass(frozen=True)
class Turn:
    """一轮动作：调用了哪个工具 + 参数 + 结果。"""

    action: str
    args: dict
    result: ToolResult


def format_observation(tool: str, result: ToolResult) -> str:
    """把 ToolResult 转成模型下一轮能读的观察 JSON（对齐 DESIGN.md §6）。

    成功：{"tool", "ok": true, "output"}
    失败：{"tool", "ok": false, "error_type", "message"}
    """
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

    def add_final(self, message: str) -> None:
        """追加 final 动作的结论（assistant 消息），让下个任务看到上一轮总结。"""
        self.messages.append(Message("assistant", json.dumps({"action": "final", "message": message}, ensure_ascii=False)))

    def append_turn(self, action: str, args: dict, result: ToolResult) -> None:
        """追加 assistant 动作（canonical）+ user 观察 + 一条结构化 Turn。"""
        assistant = json.dumps({"action": action, "args": args}, ensure_ascii=False)
        self.messages.append(Message("assistant", assistant))
        self.messages.append(Message("user", format_observation(action, result)))
        self.turns.append(Turn(action, args, result))

    def as_openai(self) -> list[dict]:
        """转成 chat 接口需要的 [{role, content}]。"""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def total_chars(self) -> int:
        """字符预算：所有消息内容长度之和，供 context 判断是否超阈值。"""
        return sum(len(m.content) for m in self.messages)
