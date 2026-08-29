"""context.py 的单元测试。

覆盖 DESIGN.md §3 context.py：
  - build(conversation) 从 conversation.turns 装配；超阈值压缩、保留最近 N 轮 + 当前任务
  - summarize 的语义压缩接入点（由上层提供 summarizer）
"""

from agent.context import RECENT_TURNS, build, summarize
from agent.messages import Conversation, Turn
from agent.tools import ToolResult


def turn(action, args, ok=True, error="", output=""):
    return Turn(action, args, ToolResult(ok=ok, output=output, error=error))


def fake_summarizer(turns):
    return " | ".join(f"{t.action}:{t.result.error or 'done'}" for t in turns)


def make_conversation(turns=(), task="task", system="rules"):
    conv = Conversation()
    conv.set_system(system)
    conv.add_task(task)
    for t in turns:
        conv.append_turn(t.action, t.args, t.result)
    return conv


# ---- build ----

def test_build_under_threshold_returns_conversation():
    conv = make_conversation([turn("list_files", {"path": "."})])
    msgs = build(conv, max_chars=100_000)
    assert msgs[0] == {"role": "system", "content": "rules"}
    assert msgs[1] == {"role": "user", "content": "task"}
    assert msgs[2]["role"] == "assistant"
    assert msgs[3]["role"] == "user"


def test_build_over_threshold_compresses():
    conv = make_conversation([
        turn("list_files", {"path": "."}),
        turn("read_file", {"path": "a.py"}),
        turn("grep", {"pattern": "foo"}),
    ])
    msgs = build(conv, recent_turns=1, max_chars=10, summarize_fn=fake_summarizer)
    contents = [m["content"] for m in msgs]
    assert any("摘要" in c for c in contents)             # 摘要出现
    assert any('"action": "grep"' in c for c in contents)  # 最近 1 轮保留
    assert not any('"action": "list_files"' in c for c in contents)  # 旧轮被取代


def test_build_over_threshold_order():
    conv = make_conversation([
        turn("list_files", {"path": "."}),
        turn("read_file", {"path": "a.py"}),
        turn("grep", {"pattern": "foo"}),
    ], task="当前任务")
    msgs = build(conv, recent_turns=1, max_chars=10, summarize_fn=fake_summarizer)
    # 顺序：system -> 摘要 -> 当前任务 -> 最近一轮(assistant + observation)
    assert msgs[0]["role"] == "system"
    assert "摘要" in msgs[1]["content"]
    assert msgs[2] == {"role": "user", "content": "当前任务"}
    assert msgs[3]["role"] == "assistant"
    assert msgs[4]["role"] == "user"  # 最后一条是 observation，不是任务
    assert msgs[-1]["content"] != "当前任务"


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
