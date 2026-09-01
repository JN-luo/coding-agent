"""上下文管理：三层上下文 + 语义压缩。

三层（DESIGN.md §3）：
  1. 固定系统规则 —— prompts.py 生成，session 启动时 set_system 一次。
  2. 工作区摘要 —— 从 conversation 的历史观察自然体现。
  3. 短期轨迹 —— conversation.turns 里最近 N 轮 action/result 原文。

历史超阈值时，把旧轨迹交给语义 summarizer 压缩，保留未完成事项和关键结论。
context 只负责从 conversation 装配和触发压缩，不做规则摘要。
"""

from collections.abc import Callable

from agent.messages import Turn, format_observation


MAX_CONTEXT_CHARS = 30_000
RECENT_TURNS = 5

SummarizeFn = Callable[[list[Turn]], str]


def build(conversation, *, recent_turns=RECENT_TURNS, max_chars=MAX_CONTEXT_CHARS, summarize_fn=None) -> list[dict]:
    """从 conversation 装配发给模型的消息，超阈值时压缩旧轨迹、保留最近 recent_turns 轮。"""
    turns = conversation.turns
    if _needs_compression(turns, recent_turns, max_chars):
        if summarize_fn is None:
            raise ValueError("需要 summarize_fn 才能进行语义压缩")
        summary = summarize(turns, recent_turns, summarize_fn)
        messages = [{"role": "system", "content": _system_content(conversation)}]
        messages.append({"role": "user", "content": "之前步骤的摘要：\n" + summary})
        if conversation.last_task:
            messages.append({"role": "user", "content": conversation.last_task})
        messages.extend(_recent_tool_messages(conversation, recent_turns))
        return messages
    return conversation.as_openai()


def summarize(turns, recent_turns, summarize_fn) -> str:
    """把旧轨迹交给语义 summarizer；无旧轮时返回 ""。"""
    old = turns[:-recent_turns] if recent_turns > 0 else turns
    if not old:
        return ""
    summary = summarize_fn(old)
    if not isinstance(summary, str) or summary.strip() == "":
        raise ValueError("summarize_fn 必须返回非空字符串")
    return summary


# ---- helpers ----

def _needs_compression(turns, recent_turns, max_chars) -> bool:
    old = turns[:-recent_turns] if recent_turns > 0 else turns
    return bool(old) and _trajectory_chars(turns) > max_chars


def _system_content(conversation) -> str:
    if conversation.messages and conversation.messages[0].role == "system":
        return conversation.messages[0].content
    return ""


def _recent_tool_messages(conversation, recent_turns) -> list[dict]:
    """保留最近工具调用的原始消息片段，避免丢失 provider-specific 字段。"""
    if recent_turns <= 0:
        return []
    messages = conversation.as_openai()
    tool_seen = 0
    start = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "tool":
            tool_seen += 1
            if tool_seen >= recent_turns:
                start = _tool_group_start(messages, i)
                break
    if start is None:
        return []
    return [
        m for m in messages[start:]
        if m.get("role") == "tool" or (m.get("role") == "assistant" and m.get("tool_calls"))
    ]


def _tool_group_start(messages, tool_index) -> int:
    for i in range(tool_index - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            return i
        if msg.get("role") in {"user", "system"}:
            break
    return tool_index


def _trajectory_chars(turns) -> int:
    return sum(len(str(t.args)) + len(format_observation(t.action, t.result)) for t in turns)
