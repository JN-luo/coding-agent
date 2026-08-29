"""messages.py 的单元测试。

覆盖 DESIGN.md §3 messages.py：
  - Message / Turn 数据结构
  - format_observation 成功/失败两种 JSON 形状（§6）
  - Conversation：set_system / add_task / append_turn / as_openai / total_chars
  - messages + turns 双轨累积，跨任务
"""

import json

from agent.messages import Conversation, Message, Turn, format_observation
from agent.tools import ToolResult


def ok(output="ok"):
    return ToolResult(ok=True, output=output)


def fail(error="E", output=""):
    return ToolResult(ok=False, output=output, error=error)


# ---- Message / Turn ----

def test_message_fields():
    m = Message("user", "hi")
    assert m.role == "user"
    assert m.content == "hi"


def test_turn_fields():
    t = Turn("read_file", {"path": "a.py"}, ok("x"))
    assert t.action == "read_file"
    assert t.args == {"path": "a.py"}
    assert t.result.ok is True


# ---- format_observation ----

def test_format_observation_success():
    obj = json.loads(format_observation("read_file", ok("x")))
    assert obj == {"tool": "read_file", "ok": True, "output": "x"}


def test_format_observation_failure():
    obj = json.loads(format_observation("read_file", fail("FileNotFound", "nope")))
    assert obj == {"tool": "read_file", "ok": False, "error_type": "FileNotFound", "message": "nope"}


# ---- Conversation ----

def test_set_system_prepends_and_replaces():
    conv = Conversation()
    conv.add_task("t1")
    conv.set_system("rules")
    assert [m.role for m in conv.messages] == ["system", "user"]
    conv.set_system("rules2")
    assert [m.role for m in conv.messages] == ["system", "user"]
    assert conv.messages[0].content == "rules2"


def test_append_turn_tracks_both():
    conv = Conversation()
    conv.add_task("t")
    conv.append_turn("list_files", {"path": "."}, ok("a.py"))
    # messages: [user task, assistant, user obs]
    assert [m.role for m in conv.messages] == ["user", "assistant", "user"]
    assert len(conv.turns) == 1
    assert conv.turns[0].action == "list_files"


def test_append_turn_assistant_is_canonical():
    conv = Conversation()
    conv.add_task("t")
    conv.append_turn("list_files", {"path": "."}, ok("a.py"))
    assistant = conv.messages[1]
    assert assistant.role == "assistant"
    assert json.loads(assistant.content) == {"action": "list_files", "args": {"path": "."}}


def test_add_final_appends_assistant():
    conv = Conversation()
    conv.add_task("t")
    conv.add_final("搞定了")
    assert [m.role for m in conv.messages] == ["user", "assistant"]
    assert json.loads(conv.messages[1].content) == {"action": "final", "message": "搞定了"}
    assert conv.turns == []  # final 不进 turns


def test_cross_task_accumulates():
    conv = Conversation()
    conv.add_task("task1")
    conv.append_turn("list_files", {}, ok("a.py"))
    conv.add_task("task2")
    conv.append_turn("read_file", {"path": "a.py"}, ok("x=1"))
    assert [m.role for m in conv.messages] == ["user", "assistant", "user", "user", "assistant", "user"]
    assert len(conv.turns) == 2
    assert conv.turns[1].action == "read_file"


def test_as_openai_and_total_chars():
    conv = Conversation()
    conv.set_system("rules")
    conv.add_task("hi")
    assert conv.as_openai() == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]
    assert conv.total_chars() == len("rules") + len("hi")
