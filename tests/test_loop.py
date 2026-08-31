"""loop.py 的单元测试。

覆盖 DESIGN.md §5/§6：
  - 五个终止条件（final / max_steps / 连续解析失败 / 同类工具失败 / 安全拒绝）
  - parse 失败回填 + 自纠错
  - 报告推导（modified_files / ran_tests / test_result / pending / stop_reason），且只统计当前任务
  - 语义压缩失败跳过压缩（非致命）
"""

import json

from agent.llm import LLMError
from agent.loop import Report, build_report, render_report, run_task
from agent.messages import Turn
from agent.session import Session
from agent.tools import ToolResult


def j(action, args=None):
    d = {"action": action}
    if args is not None:
        d["args"] = args
    return json.dumps(d)


def jfinal(msg="done"):
    return json.dumps({"action": "final", "message": msg})


class FakeLLM:
    def __init__(self, responses, summarize_raises=False):
        self.responses = list(responses)
        self.calls = []
        self.summarize_raises = summarize_raises

    def complete(self, messages):
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


def test_max_steps(tmp_path):
    s = session(tmp_path, [j("list_files", {"path": "."})], max_steps=3)
    r = run_task(s, "任务")
    assert r.done is False
    assert r.stop_reason == "max_steps"
    assert r.steps == 3


def test_parse_failures(tmp_path):
    s = session(tmp_path, ["这不是 json"])
    r = run_task(s, "任务")
    assert r.stop_reason == "parse_failures"
    assert r.done is False


def test_parse_failure_then_recover(tmp_path):
    s = session(tmp_path, ["这不是 json", j("list_files", {"path": "."}), jfinal("ok")])
    r = run_task(s, "任务")
    assert r.stop_reason == "final"
    assert r.done is True


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
    assert events == ["action", "tool", "action"]


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


def test_ask_mode_grants_after_first_yes(tmp_path):
    asks = []

    def asker(action, args):
        asks.append((action, args))
        return True

    responses = [
        j("write_file", {"path": "a.py", "content": "x"}),
        j("write_file", {"path": "b.py", "content": "y"}),
        jfinal("ok"),
    ]
    s = Session(FakeLLM(responses), tmp_path, run_tool_fn=ok_run_tool, mode="ask", asker=asker)
    r = run_task(s, "任务")
    assert r.stop_reason == "final"
    assert len(asks) == 1  # 同类 write_file 第二次免问
    assert len([t for t in s.conversation.turns if t.action == "write_file"]) == 2


def test_ask_mode_grants_do_not_cross_tasks(tmp_path):
    asks = []

    def asker(action, args):
        asks.append((action, args))
        return True

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
        return True

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
                run_tool_fn=ok_run_tool, mode="ask", asker=lambda a, args: False)
    r = run_task(s, "任务")
    assert r.stop_reason == "final"
    write_turns = [t for t in s.conversation.turns if t.action == "write_file"]
    assert write_turns and write_turns[0].result.error == "UserDenied"


def test_ask_mode_denies_same_action_without_asking_again(tmp_path):
    asks = []

    def asker(action, args):
        asks.append((action, args))
        return False

    responses = [
        j("write_file", {"path": "a.py", "content": "x"}),
        j("write_file", {"path": "b.py", "content": "y"}),
        jfinal("ok"),
    ]
    s = Session(FakeLLM(responses), tmp_path, run_tool_fn=ok_run_tool, mode="ask", asker=asker)
    r = run_task(s, "任务")
    assert r.stop_reason == "user_denied"
    assert len(asks) == 1
    write_turns = [t for t in s.conversation.turns if t.action == "write_file"]
    assert len(write_turns) == 2
    assert all(t.result.error == "UserDenied" for t in write_turns)


def test_ask_mode_user_denied_stop_after_two_denials(tmp_path):
    responses = [
        j("write_file", {"path": "a.py", "content": "x"}),
        j("run_command", {"command": "pytest -q"}),
        jfinal("ok"),
    ]
    s = Session(FakeLLM(responses), tmp_path, run_tool_fn=ok_run_tool, mode="ask", asker=lambda a, args: False)
    r = run_task(s, "任务")
    assert r.stop_reason == "user_denied"
    assert r.done is False


def test_ask_mode_denies_do_not_cross_tasks(tmp_path):
    asks = []

    def asker(action, args):
        asks.append((action, args))
        return False

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
