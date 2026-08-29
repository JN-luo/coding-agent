"""单任务执行器（Observe -> Think -> Act -> Observe），由 Session 驱动（DESIGN.md §2/§5）。

只负责完成一个 task 的推理闭环，写回 session.conversation（活上下文）与
session.trace（仅日志）。跨任务状态（conversation / trace / workspace）归 Session；
单任务状态（step / 各类失败计数）在本函数内局部，每任务重置。
"""

import json
from dataclasses import dataclass

from agent.context import build
from agent.llm import LLMError
from agent.parser import ParseError, parse
from agent.tools import TOOLS, ToolResult

MAX_STEPS = 20
MAX_PARSE_FAILURES = 3
MAX_SAME_TOOL_FAILURES = 3
MAX_SAFETY_REJECTIONS = 3

SAFETY_ERRORS = {"PolicyDenied", "PathOutsideWorkspace"}

# 压缩失败时禁用压缩的阈值（大到永不触发）
_NEVER_COMPRESS = 10**12


@dataclass(frozen=True)
class Report:
    done: bool
    message: str
    stop_reason: str = ""
    modified_files: tuple[str, ...] = ()
    ran_tests: bool = False
    test_result: str = ""
    pending: tuple[str, ...] = ()
    steps: int = 0


def run_task(session, task: str) -> Report:
    """完成一个 task 的推理闭环，写回 session.conversation + session.trace。"""
    session.trace.log("task", text=task)
    session.conversation.add_task(task)
    start_turns = len(session.conversation.turns)

    step = 0
    parse_failures = 0
    same_failures = 0
    last_fp = None
    safety = 0

    while step < session.max_steps:
        step += 1
        messages = _build_messages(session)
        try:
            raw = session.llm.complete(messages)
        except LLMError as exc:
            return _report(session, False, str(exc), "llm_error", start_turns, step)
        session.trace.log("llm_raw", step=step, text=raw)

        try:
            action = parse(raw, frozenset(TOOLS))
        except ParseError as exc:
            parse_failures += 1
            same_failures = 0
            session.trace.log("parse_error", step=step, err=str(exc))
            session.conversation.append_turn(
                "parse_error", {},
                ToolResult(ok=False, output=f"输出无法解析：{exc}，请重新只输出一个 JSON 对象。", error="ParseError"),
            )
            if parse_failures >= MAX_PARSE_FAILURES:
                return _report(session, False, "连续解析失败", "parse_failures", start_turns, step)
            continue

        parse_failures = 0
        session.trace.log("action", step=step, action=action.action, args=action.args)

        if action.action == "final":
            session.conversation.add_final(action.message)
            return _report(session, True, action.message, "final", start_turns, step)

        result = session.run_tool_fn(action.action, action.args, session.workspace)
        session.trace.log("tool", step=step, tool=action.action, ok=result.ok,
                          error=result.error, output_len=len(result.output))
        session.conversation.append_turn(action.action, action.args, result)

        if result.ok:
            same_failures = 0
            last_fp = None
        else:
            if result.error in SAFETY_ERRORS:
                safety += 1
                if safety >= MAX_SAFETY_REJECTIONS:
                    return _report(session, False, "多次越权或危险命令", "safety_rejections", start_turns, step)
            fp = _fingerprint(action.action, action.args)
            same_failures = same_failures + 1 if fp == last_fp else 1
            last_fp = fp
            if same_failures >= MAX_SAME_TOOL_FAILURES:
                return _report(session, False, "连续同类工具失败", "same_tool_failures", start_turns, step)

    return _report(session, False, "达到最大步数", "max_steps", start_turns, step)


# ---- 内部 ----

def _build_messages(session):
    try:
        return build(session.conversation,
                     recent_turns=session.recent_turns, max_chars=session.max_chars,
                     summarize_fn=session.llm.summarize)
    except (LLMError, ValueError) as exc:
        # 语义压缩失败非致命：记日志，跳过压缩（阈值拉满，保留完整 conversation）
        session.trace.log("summarize_error", err=str(exc), turns=len(session.conversation.turns))
        return build(session.conversation,
                     recent_turns=session.recent_turns, max_chars=_NEVER_COMPRESS)


def _report(session, done, message, stop_reason, start_turns, step):
    task_turns = session.conversation.turns[start_turns:]
    report = build_report(task_turns, done=done, message=message, stop_reason=stop_reason, steps=step)
    session.trace.log("stop", reason=stop_reason, steps=report.steps, done=done)
    return report


# ---- 报告 ----

def build_report(turns, *, done, message, stop_reason, steps):
    modified = tuple(
        t.args["path"] for t in turns
        if t.action == "write_file" and t.result.ok and "path" in t.args
    )
    test_turns = [
        t for t in turns
        if t.action == "run_command" and "test" in t.args.get("command", "").lower()
    ]
    test_result = _clip(test_turns[-1].result.output, 2000) if test_turns else ""
    pending = tuple(_describe_turn(t) for t in turns if not t.result.ok and t.action != "parse_error")
    return Report(
        done=done, message=message, stop_reason=stop_reason,
        modified_files=modified, ran_tests=bool(test_turns), test_result=test_result,
        pending=pending, steps=steps,
    )


def render_report(report: Report) -> str:
    lines = []
    if report.done:
        lines.append(f"完成：{report.message}")
    else:
        lines.append(f"未完成（{report.stop_reason}）：{report.message}")
    lines.append(f"步骤数：{report.steps}")
    if report.modified_files:
        lines.append("修改的文件：")
        lines += [f"  - {f}" for f in report.modified_files]
    lines.append(f"是否运行测试：{'是' if report.ran_tests else '否'}")
    if report.test_result:
        lines.append("测试结果：")
        lines.append(report.test_result)
    if report.pending:
        lines.append("未完成事项：")
        lines += [f"  - {p}" for p in report.pending]
    return "\n".join(lines)


# ---- helpers ----

def _fingerprint(action, args):
    return (action, json.dumps(args, sort_keys=True, ensure_ascii=False))


def _label(turn):
    args = ", ".join(f"{k}={v}" for k, v in turn.args.items())
    return f"{turn.action}({args})" if args else turn.action


def _describe_turn(turn):
    return f"{_label(turn)}: {turn.result.error}"


def _clip(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
