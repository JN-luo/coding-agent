"""跨任务状态与编排（Session）。

一个进程内长期存在的会话：持有 Conversation（活上下文）、Trace（仅日志）、
workspace、llm、以及各任务的 Report。REPL 里一个 session 跑到底，退出才销毁。
"""

from agent.context import MAX_CONTEXT_CHARS, RECENT_TURNS
from agent.loop import MAX_STEPS, Report, run_task
from agent.messages import Conversation
from agent.tools import run_tool
from agent.trace import NullTracer


class Session:
    def __init__(
        self,
        llm,
        workspace,
        *,
        trace=None,
        run_tool_fn=run_tool,
        max_steps: int = MAX_STEPS,
        recent_turns: int = RECENT_TURNS,
        max_chars: int = MAX_CONTEXT_CHARS,
    ):
        self.llm = llm
        self.workspace = workspace
        self.trace = trace or NullTracer()
        self.run_tool_fn = run_tool_fn
        self.max_steps = max_steps
        self.recent_turns = recent_turns
        self.max_chars = max_chars
        self.conversation = Conversation()
        self.reports: list[Report] = []

    def set_system(self, rules: str) -> None:
        self.conversation.set_system(rules)

    def submit(self, task: str) -> Report:
        report = run_task(self, task)
        self.reports.append(report)
        return report
