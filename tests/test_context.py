"""context.py 的单元测试。

覆盖 DESIGN.md §3 context.py：
  - build(conversation) 从 conversation.turns 装配；超阈值压缩、保留最近 N 轮 + 当前任务
  - summarize 的语义压缩接入点（由上层提供 summarizer）
"""

import json

from agent.context import RECENT_TURNS, build, summarize
from agent.messages import Conversation, Turn
from agent.tools import ToolResult


def turn(action, args, ok=True, error="", output=""):
    return Turn(action, args, ToolResult(ok=ok, output=output, error=error))


def fake_summarizer(turns):
    return " | ".join(f"{t.action}:{t.result.error or 'done'}" for t in turns)


class _TC:
    def __init__(self, id, name, arguments):
        self.id = id
        self.name = name
        self.arguments = arguments
        self.raw_arguments = json.dumps(arguments, ensure_ascii=False)
        self.parse_error = ""


class _Response:
    def __init__(self, tool_calls, reasoning_content=None):
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


def make_conversation(turns=(), task="task", system="rules"):
    conv = Conversation()
    conv.set_system(system)
    conv.add_task(task)
    for i, t in enumerate(turns):
        conv.append_assistant_tool_calls(_Response([_TC(f"call_{i}", t.action, t.args)], reasoning_content=f"think-{i}"))
        conv.append_tool_result(f"call_{i}", t.action, t.args, t.result)
    return conv


# ---- build ----

def test_build_under_threshold_returns_conversation():
    conv = make_conversation([turn("list_files", {"path": "."})])
    msgs = build(conv, max_chars=100_000)
    assert msgs[0] == {"role": "system", "content": "rules"}
    assert msgs[1] == {"role": "user", "content": "task"}
    assert msgs[2]["role"] == "assistant"
    assert "tool_calls" in msgs[2]
    assert msgs[3]["role"] == "tool"


def test_build_over_threshold_compresses():
    conv = make_conversation([
        turn("list_files", {"path": "."}),
        turn("read_file", {"path": "a.py"}),
        turn("grep", {"pattern": "foo"}),
    ])
    msgs = build(conv, recent_turns=1, max_chars=10, summarize_fn=fake_summarizer)
    assert any("摘要" in (m.get("content") or "") for m in msgs)  # 摘要出现
    # 最近 1 轮（grep）保留为 tool_calls；旧轮（list_files）被取代
    names = [m["tool_calls"][0]["function"]["name"] for m in msgs if m.get("tool_calls")]
    assert names == ["grep"]


def test_build_over_threshold_order():
    conv = make_conversation([
        turn("list_files", {"path": "."}),
        turn("read_file", {"path": "a.py"}),
        turn("grep", {"pattern": "foo"}),
    ], task="当前任务")
    msgs = build(conv, recent_turns=1, max_chars=10, summarize_fn=fake_summarizer)
    # 顺序：system -> 摘要 -> 当前任务 -> 最近一轮(assistant tool_calls + tool 结果)
    assert msgs[0]["role"] == "system"
    assert "摘要" in msgs[1]["content"]
    assert msgs[2] == {"role": "user", "content": "当前任务"}
    assert msgs[3]["role"] == "assistant"
    assert "tool_calls" in msgs[3]
    assert msgs[4]["role"] == "tool"
    assert msgs[4]["name"] == "grep"
    assert msgs[3]["reasoning_content"] == "think-2"


# ---- summarize ----

def test_summarize_empty():
    assert summarize([], RECENT_TURNS, fake_summarizer) == ""


def test_summarize_nothing_old():
    assert summarize([turn("list_files", {})], RECENT_TURNS, fake_summarizer) == ""


def test_summarize_uses_summarizer():
    turns = [turn("read_file", {"path": "a.py"})]
    assert summarize(turns, 0, fake_summarizer) == "read_file:done"


def test_summarize_keeps_recent_out():
    turns = [
        turn("list_files", {"path": "."}),
        turn("read_file", {"path": "a.py"}),
        turn("grep", {"pattern": "foo"}),
    ]
    s = summarize(turns, 1, fake_summarizer)
    assert "list_files" in s
    assert "read_file" in s
    assert "grep" not in s
