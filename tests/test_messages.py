"""messages.py 的单元测试。

覆盖 DESIGN.md §3 messages.py：
  - Message / Turn 数据结构
  - format_observation 成功/失败两种 JSON 形状（§6）
  - Conversation：set_system / add_task / append_assistant_tool_calls / append_tool_result / add_final
  - as_openai 发射 tool_calls / tool_call_id 形状
"""

import json

from agent.messages import Conversation, Message, Turn, format_observation
from agent.tools import ToolResult


def ok(output="ok"):
    return ToolResult(ok=True, output=output)


def fail(error="E", output=""):
    return ToolResult(ok=False, output=output, error=error)


class TC:
    def __init__(self, id="call_1", name="list_files", arguments=None, raw_arguments="{}"):
        self.id = id
        self.name = name
        self.arguments = arguments or {}
        self.raw_arguments = raw_arguments
        self.parse_error = ""


class Response:
    def __init__(self, tool_calls, reasoning_content=None):
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


# ---- Message / Turn ----

def test_message_fields():
    m = Message("user", "hi")
    assert m.role == "user"
    assert m.content == "hi"
    assert m.name is None


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
    assert conv.messages[0].content == "rules2"


def test_append_tool_result_tracks_both():
    conv = Conversation()
    conv.add_task("t")
    conv.append_tool_result("call_1", "list_files", {"path": "."}, ok("a.py"))
    assert [m.role for m in conv.messages] == ["user", "tool"]
    assert len(conv.turns) == 1
    assert conv.turns[0].action == "list_files"


def test_add_final_appends_plain_assistant():
    conv = Conversation()
    conv.add_task("t")
    conv.add_final("搞定了")
    assert [m.role for m in conv.messages] == ["user", "assistant"]
    assert conv.messages[1].content == "搞定了"


def test_append_assistant_tool_calls():
    conv = Conversation()
    conv.add_task("t")
    conv.append_assistant_tool_calls(Response([TC(arguments={"path": "."})]))
    m = conv.messages[-1]
    assert m.role == "assistant"
    assert m.tool_calls == [{"id": "call_1", "type": "function", "function": {"name": "list_files", "arguments": '{"path": "."}'}}]


def test_append_assistant_tool_calls_preserves_raw_arguments():
    conv = Conversation()
    conv.add_task("t")
    conv.append_assistant_tool_calls(Response([TC(arguments={}, raw_arguments='{"path":')]))
    assert conv.messages[-1].tool_calls[0]["function"]["arguments"] == '{"path":'


def test_append_assistant_tool_calls_preserves_reasoning_content():
    conv = Conversation()
    conv.add_task("t")
    conv.append_assistant_tool_calls(Response([TC()], reasoning_content="thinking"))
    msgs = conv.as_openai()
    assert msgs[-1]["reasoning_content"] == "thinking"


def test_as_openai_emits_tool_shape():
    conv = Conversation()
    conv.set_system("rules")
    conv.add_task("hi")
    conv.append_assistant_tool_calls(Response([TC()]))
    conv.append_tool_result("call_1", "list_files", {"path": "."}, ok("a.py"))
    msgs = conv.as_openai()
    assert msgs[0] == {"role": "system", "content": "rules"}
    assert msgs[1] == {"role": "user", "content": "hi"}
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] is None
    assert "tool_calls" in msgs[2]
    assert msgs[3]["role"] == "tool"
    assert msgs[3]["tool_call_id"] == "call_1"
    assert msgs[3]["name"] == "list_files"


def test_total_chars():
    conv = Conversation()
    conv.add_task("ab")   # 2
    conv.add_final("cd")  # 2
    assert conv.total_chars() == 4
