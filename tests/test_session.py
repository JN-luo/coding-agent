"""session.py 的单元测试。

覆盖 DESIGN.md §3 session.py：
  - submit 逐任务返回 Report 并聚合到 reports
  - conversation 跨任务累积（messages + turns 双轨）
"""

import json

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
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, messages):
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    def summarize(self, turns):
        return "已完成：...\n仍需：..."


def ok_run_tool(action, args, workspace):
    return ToolResult(ok=True, output="ok")


def test_submit_aggregates_reports(tmp_path):
    s = Session(FakeLLM([jfinal("t1"), jfinal("t2")]), tmp_path, run_tool_fn=ok_run_tool)
    r1 = s.submit("任务1")
    r2 = s.submit("任务2")
    assert s.reports == [r1, r2]
    assert r1.message == "t1"
    assert r2.message == "t2"


def test_conversation_accumulates_across_tasks(tmp_path):
    responses = [
        j("list_files", {"path": "."}), jfinal("t1"),
        j("read_file", {"path": "a.py"}), jfinal("t2"),
    ]
    s = Session(FakeLLM(responses), tmp_path, run_tool_fn=ok_run_tool)
    s.submit("任务1")
    s.submit("任务2")
    assert len(s.conversation.turns) == 2
    assert s.conversation.last_task == "任务2"
    # 无 system：任务1 + (a1,o1) + final1 + 任务2 + (a2,o2) + final2 = 8 条消息
    assert len(s.conversation.messages) == 8
