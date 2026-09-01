"""loop.py 的单元测试。

覆盖 DESIGN.md §5/§6：
  - 五个终止条件（final / max_steps / 连续解析失败 / 同类工具失败 / 安全拒绝）
  - parse 失败回填 + 自纠错
  - 报告推导（modified_files / ran_tests / test_result / pending / stop_reason），且只统计当前任务
  - 语义压缩失败跳过压缩（非致命）
"""

import json

from agent.llm import LLMError, ModelResponse, ModelToolCall
from agent.loop import Report, build_report, render_report, run_task
from agent.messages import Turn
from agent.policy import ASK_DENY, ASK_ONCE, ASK_REMEMBER
from agent.session import Session
from agent.tools import ToolResult


def j(action, args=None):
    return ModelResponse(tool_calls=[ModelToolCall(id="call_1", name=action, arguments=args or {})])


def jbad_args(action):
    return ModelResponse(tool_calls=[
        ModelToolCall(id="call_1", name=action, arguments={}, raw_arguments='{"path":', parse_error="bad json")
    ])


def jfinal(msg="done"):
    return ModelResponse(content=msg)


class FakeLLM:
    def __init__(self, responses, summarize_raises=False):
        self.responses = list(responses)
        self.calls = []
        self.summarize_raises = summarize_raises

    def complete(self, messages, tools=None):
        self.calls.append(messages)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    def summarize(self, turns):
        if self.summarize_raises:
            raise LLMError("summarize failed")
        return "已完成：...\n仍需：..."


class FakeTracer:
    def __init__(self):
        self.events = []

    def log(self, event, **fields):
        self.events.append((event, fields))


def ok_run_tool(action, args, workspace):
    return ToolResult(ok=True, output="ok")


def fail_run_tool(action, args, workspace):
    return ToolResult(ok=False, output="", error="CommandFailed")


def session(tmp_path, responses, run_tool_fn=ok_run_tool, **kw):
    return Session(FakeLLM(responses), tmp_path, run_tool_fn=run_tool_fn, **kw)


# ---- 终止条件 ----

def test_final_action(tmp_path):
    s = session(tmp_path, [jfinal("搞定了")])
    r = run_task(s, "任务")
    assert r.done is True
    assert r.stop_reason == "final"
    assert r.message == "搞定了"
    assert r.steps == 1


def test_empty_response_is_not_final(tmp_path):
    s = session(tmp_path, [ModelResponse(content="")])
    r = run_task(s, "任务")
    assert r.done is False
    assert r.stop_reason == "llm_error"


def test_max_steps(tmp_path):
    s = session(tmp_path, [j("list_files", {"path": "."})], max_steps=3)
    r = run_task(s, "任务")
    assert r.done is False
    assert r.stop_reason == "max_steps"
    assert r.steps == 3


def test_same_tool_failure(tmp_path):
    cmd = j("run_command", {"command": "python -m pytest"})
    s = session(tmp_path, [cmd], run_tool_fn=fail_run_tool, mode="auto")
    r = run_task(s, "任务")
    assert r.stop_reason == "same_tool_failures"


def test_safety_rejection(tmp_path):
    s = session(tmp_path, [j("read_file", {"path": ".env"})],
                run_tool_fn=lambda action, args, ws: ToolResult(ok=False, output="", error="PolicyDenied"))
    r = run_task(s, "任务")
    assert r.stop_reason == "safety_rejections"


def test_on_step_callback(tmp_path):
    events = []

    def on_step(kind, **fields):
        events.append(kind)

    s = Session(FakeLLM([j("list_files", {"path": "."}), jfinal("ok")]), tmp_path,
                run_tool_fn=ok_run_tool, on_step=on_step)
    run_task(s, "任务")
    assert events == ["action", "tool"]


def test_on_step_callback_uses_substeps_for_multiple_tool_calls(tmp_path):
    labels = []

    def on_step(kind, **fields):
        if kind == "action":
            labels.append((fields["step"], fields.get("substep"), fields["action"]))

    response = ModelResponse(tool_calls=[
        ModelToolCall(id="call_1", name="list_files", arguments={"path": "."}),
        ModelToolCall(id="call_2", name="read_file", arguments={"path": "a.py"}),
    ])
    s = Session(FakeLLM([response, jfinal("ok")]), tmp_path, run_tool_fn=ok_run_tool, on_step=on_step)
    run_task(s, "任务")
    assert labels == [(1, 1, "list_files"), (1, 2, "read_file")]


def test_tool_call_argument_parse_error_becomes_observation(tmp_path):
    s = session(tmp_path, [jbad_args("read_file"), jfinal("ok")])
    r = run_task(s, "任务")
    assert r.stop_reason == "final"
    turn = s.conversation.turns[0]
    assert turn.action == "read_file"
    assert turn.result.error == "InvalidArgs"
    assert "bad json" in turn.result.output


def test_trace_records_failed_output_preview(tmp_path):
    tracer = FakeTracer()

    def boom(action, args, workspace):
        return ToolResult(ok=False, output="x" * 2000, error="CommandFailed")

    s = Session(FakeLLM([j("run_command", {"command": "pytest -q"}), jfinal("ok")]), tmp_path,
                trace=tracer, run_tool_fn=boom, mode="auto")
    run_task(s, "任务")
    tool_events = [fields for event, fields in tracer.events if event == "tool"]
    assert tool_events[0]["output_preview"].endswith("...")
    assert len(tool_events[0]["output_preview"]) < 1200


def test_auto_mode_allows_write(tmp_path):
    s = Session(FakeLLM([j("write_file", {"path": "a.py", "content": "x"}), jfinal("ok")]), tmp_path,
                run_tool_fn=ok_run_tool, mode="auto")
    r = run_task(s, "任务")
    assert r.stop_reason == "final"
    assert any(t.action == "write_file" and t.result.ok for t in s.conversation.turns)


def test_readonly_mode_denies_write(tmp_path):
    s = Session(FakeLLM([j("write_file", {"path": "a.py", "content": "x"}), jfinal("ok")]), tmp_path,
                run_tool_fn=ok_run_tool, mode="readonly")
    r = run_task(s, "任务")
    assert r.stop_reason == "final"
    write_turns = [t for t in s.conversation.turns if t.action == "write_file"]
    assert write_turns and write_turns[0].result.error == "PolicyDenied"


def test_ask_mode_once_allows_only_current_call(tmp_path):
    asks = []

    def asker(action, args):
        asks.append((action, args))
        return ASK_ONCE

    responses = [
        j("write_file", {"path": "a.py", "content": "x"}),
        j("write_file", {"path": "b.py", "content": "y"}),
        jfinal("ok"),
    ]
    s = Session(FakeLLM(responses), tmp_path, run_tool_fn=ok_run_tool, mode="ask", asker=asker)
    r = run_task(s, "任务")
    assert r.stop_reason == "final"
    assert len(asks) == 2
    assert len([t for t in s.conversation.turns if t.action == "write_file"]) == 2


def test_ask_mode_remember_write_allows_later_writes(tmp_path):
    asks = []

    def asker(action, args):
        asks.append((action, args))
        return ASK_REMEMBER

    responses = [
        j("write_file", {"path": "a.py", "content": "x"}),
        j("write_file", {"path": "b.py", "content": "y"}),
        jfinal("ok"),
    ]
    s = Session(FakeLLM(responses), tmp_path, run_tool_fn=ok_run_tool, mode="ask", asker=asker)
    r = run_task(s, "任务")
    assert r.stop_reason == "final"
    assert len(asks) == 1
    assert len([t for t in s.conversation.turns if t.action == "write_file" and t.result.ok]) == 2


def test_ask_mode_grants_do_not_cross_tasks(tmp_path):
    asks = []

    def asker(action, args):
        asks.append((action, args))
        return ASK_REMEMBER

    responses = [
        j("write_file", {"path": "a.py", "content": "x"}),
        jfinal("t1"),
        j("write_file", {"path": "b.py", "content": "y"}),
        jfinal("t2"),
    ]
    s = Session(FakeLLM(responses), tmp_path, run_tool_fn=ok_run_tool, mode="ask", asker=asker)
    s.submit("任务1")
    s.submit("任务2")
    assert [action for action, _ in asks] == ["write_file", "write_file"]


def test_ask_mode_grants_run_command_by_exact_command(tmp_path):
    asks = []

    def asker(action, args):
        asks.append(args["command"])
        return ASK_REMEMBER

    responses = [
        j("run_command", {"command": "pytest -q"}),
        j("run_command", {"command": "pytest -q"}),
        j("run_command", {"command": "pytest -q test_calculator.py"}),
        jfinal("ok"),
    ]
    s = Session(FakeLLM(responses), tmp_path, run_tool_fn=ok_run_tool, mode="ask", asker=asker)
    r = run_task(s, "任务")
    assert r.stop_reason == "final"
    assert asks == ["pytest -q", "pytest -q test_calculator.py"]


def test_ask_mode_denies_on_no(tmp_path):
    s = Session(FakeLLM([j("write_file", {"path": "a.py", "content": "x"}), jfinal("ok")]), tmp_path,
                run_tool_fn=ok_run_tool, mode="ask", asker=lambda a, args: ASK_DENY)
    r = run_task(s, "任务")
    assert r.stop_reason == "final"
    write_turns = [t for t in s.conversation.turns if t.action == "write_file"]
    assert write_turns and write_turns[0].result.error == "UserDenied"


def test_ask_mode_deny_does_not_block_later_write(tmp_path):
    asks = []
    choices = iter([ASK_DENY, ASK_ONCE])

    def asker(action, args):
        asks.append((action, args))
        return next(choices)

    responses = [
        j("write_file", {"path": "a.py", "content": "x"}),
        j("write_file", {"path": "b.py", "content": "y"}),
        jfinal("ok"),
    ]
    s = Session(FakeLLM(responses), tmp_path, run_tool_fn=ok_run_tool, mode="ask", asker=asker)
    r = run_task(s, "任务")
    assert r.stop_reason == "final"
    assert len(asks) == 2
    write_turns = [t for t in s.conversation.turns if t.action == "write_file"]
    assert write_turns[0].result.error == "UserDenied"
    assert write_turns[1].result.ok


def test_ask_mode_stops_after_three_consecutive_denials(tmp_path):
    responses = [
        j("write_file", {"path": "a.py", "content": "x"}),
        j("run_command", {"command": "pytest -q"}),
        j("write_file", {"path": "b.py", "content": "y"}),
        jfinal("ok"),
    ]
    s = Session(FakeLLM(responses), tmp_path, run_tool_fn=ok_run_tool, mode="ask",
                asker=lambda a, args: ASK_DENY)
    r = run_task(s, "任务")
    assert r.stop_reason == "user_denied"
    assert r.done is False
    assert len(s.conversation.turns) == 3


def test_ask_mode_stops_after_three_command_denials(tmp_path):
    responses = [
        j("run_command", {"command": f"pytest -q -k test_{i}"}) for i in range(5)
    ] + [jfinal("ok")]
    s = Session(FakeLLM(responses), tmp_path, run_tool_fn=ok_run_tool, mode="ask", asker=lambda a, args: ASK_DENY)
    r = run_task(s, "任务")
    assert r.stop_reason == "user_denied"
    assert r.done is False


def test_ask_mode_denies_do_not_cross_tasks(tmp_path):
    asks = []

    def asker(action, args):
        asks.append((action, args))
        return ASK_DENY

    responses = [
        j("write_file", {"path": "a.py", "content": "x"}),
        jfinal("t1"),
        j("write_file", {"path": "b.py", "content": "y"}),
        jfinal("t2"),
    ]
    s = Session(FakeLLM(responses), tmp_path, run_tool_fn=ok_run_tool, mode="ask", asker=asker)
    s.submit("任务1")
    s.submit("任务2")
    assert [action for action, _ in asks] == ["write_file", "write_file"]


def test_different_failures_do_not_stack(tmp_path):
    responses = [
        j("run_command", {"command": "python -m pytest"}),
        j("run_command", {"command": "python -m unittest"}),
        jfinal("ok"),
    ]
    s = session(tmp_path, responses, run_tool_fn=fail_run_tool, mode="auto")
    r = run_task(s, "任务")
    assert r.stop_reason == "final"


# ---- 报告只统计当前任务 ----

def test_report_scoped_to_current_task(tmp_path):
    responses = [
        j("write_file", {"path": "a.py", "content": "x"}), jfinal("t1"),
        j("write_file", {"path": "b.py", "content": "y"}), jfinal("t2"),
    ]
    s = Session(FakeLLM(responses), tmp_path, run_tool_fn=ok_run_tool, mode="auto")
    r1 = s.submit("任务1")
    r2 = s.submit("任务2")
    assert r1.modified_files == ("a.py",)
    assert r2.modified_files == ("b.py",)  # 不包含上一任务的 a.py


# ---- 报告 ----

def test_build_report_derives_fields():
    turns = [
        Turn("write_file", {"path": "a.py", "content": "x=1"}, ToolResult(ok=True, output="created: a.py")),
        Turn("run_command", {"command": "python -m pytest"}, ToolResult(ok=True, output="1 passed in 0.5s")),
        Turn("run_command", {"command": "python -c 1/0"}, ToolResult(ok=False, output="", error="CommandFailed")),
    ]
    r = build_report(turns, done=False, message="m", stop_reason="max_steps", steps=3)
    assert r.modified_files == ("a.py",)
    assert r.ran_tests is True
    assert r.test_result == "1 passed in 0.5s"
    assert any("run_command" in p and "CommandFailed" in p for p in r.pending)


def test_build_report_prefers_successful_test_result():
    turns = [
        Turn("run_command", {"command": "mvn test"}, ToolResult(ok=True, output="BUILD SUCCESS")),
        Turn("run_command", {"command": "mvn -q clean test"}, ToolResult(ok=False, output="command denied", error="PolicyDenied")),
    ]
    r = build_report(turns, done=False, message="m", stop_reason="safety_rejections", steps=2)
    assert r.ran_tests is True
    assert r.test_result == "BUILD SUCCESS"


def test_render_report(tmp_path):
    r = Report(done=False, message="达到最大步数", stop_reason="max_steps",
               modified_files=("a.py",), ran_tests=True, test_result="1 passed",
               pending=("run_command(...): CommandFailed",), steps=20)
    text = render_report(r)
    assert "max_steps" in text
    assert "a.py" in text
    assert "是" in text
    assert "CommandFailed" in text


# ---- 语义压缩失败非致命 ----

def test_summarize_failure_falls_back_and_logs(tmp_path):
    tracer = FakeTracer()
    s = Session(
        FakeLLM([j("list_files", {"path": "."}), j("read_file", {"path": "a.py"}), jfinal("ok")], summarize_raises=True),
        tmp_path,
        trace=tracer,
        run_tool_fn=ok_run_tool,
        recent_turns=1,
        max_chars=10,
    )
    r = run_task(s, "任务")
    assert r.stop_reason == "final"
    assert r.done is True
    assert any(event == "summarize_error" for event, _ in tracer.events)
