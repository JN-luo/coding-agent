"""单任务执行器（Observe -> Think -> Act -> Observe），由 Session 驱动（DESIGN.md §2/§5）。

只负责完成一个 task 的推理闭环，写回 session.conversation（活上下文）与
session.trace（仅日志）。跨任务状态（conversation / trace / workspace）归 Session；
单任务状态（step / 各类失败计数）在本函数内局部，每任务重置。
"""

import json
from dataclasses import dataclass

from agent.context import build
from agent.llm import LLMError
from agent.policy import ASK_DENY, ASK_ONCE, ASK_REMEMBER, decide, grant_key, normalize_ask_choice
from agent.tools import TOOLS, ToolResult

MAX_STEPS = 50
MAX_SAME_TOOL_FAILURES = 3
MAX_SAFETY_REJECTIONS = 3
MAX_USER_DENIALS = 5
MAX_CONSECUTIVE_USER_DENIALS = 3
MAX_USER_DENIALS_BY_ACTION = 3

SAFETY_ERRORS = {"PolicyDenied", "PathOutsideWorkspace"}

# 压缩失败时禁用压缩的阈值（大到永不触发）
_NEVER_COMPRESS = 10**12
TRACE_OUTPUT_PREVIEW = 1000


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
    task_grants: set[str] = set()
    tools_schema = [t.to_schema() for t in TOOLS.values()]

    session.trace.log("task", text=task)
    session.conversation.add_task(task)
    start_turns = len(session.conversation.turns)

    step = 0
    same_failures = 0
    last_fp = None
    safety = 0
    user_denies_total = 0
    user_denies_consecutive = 0
    user_denies_by_action = {"write_file": 0, "run_command": 0}

    while step < session.max_steps:
        step += 1
        messages = _build_messages(session)
        try:
            response = session.llm.complete(messages, tools=tools_schema)
        except LLMError as exc:
            return _report(session, False, str(exc), "llm_error", start_turns, step)
        session.trace.log("llm_response", step=step, n_tool_calls=len(response.tool_calls))

        # 无 tool_calls → final 回答
        if not response.tool_calls:
            content = response.content or ""
            if content.strip() == "":
                return _report(session, False, "模型没有返回工具调用或最终回答。", "llm_error", start_turns, step)
            session.conversation.add_final(content)
            return _report(session, True, content, "final", start_turns, step)

        # 有 tool_calls → 逐个裁决 + 执行
        session.conversation.append_assistant_tool_calls(response)
        for index, tc in enumerate(response.tool_calls, start=1):
            if session.on_step:
                session.on_step(
                    "action",
                    step=step,
                    substep=index if len(response.tool_calls) > 1 else None,
                    action=tc.name,
                    args=tc.arguments,
                )
            if tc.parse_error:
                result = ToolResult(ok=False, output=tc.parse_error, error="InvalidArgs")
            elif tc.name not in TOOLS:
                result = ToolResult(ok=False, output=f"未知工具: {tc.name}", error="UnknownTool")
            else:
                result = _execute_action(session, tc.name, tc.arguments, task_grants)
            session.conversation.append_tool_result(tc.id, tc.name, tc.arguments, result)
            trace_fields = {
                "step": step,
                "tool": tc.name,
                "ok": result.ok,
                "error": result.error,
                "output_len": len(result.output),
            }
            if not result.ok and result.output:
                trace_fields["output_preview"] = _clip(result.output, TRACE_OUTPUT_PREVIEW)
            session.trace.log("tool", **trace_fields)
            if session.on_step:
                session.on_step("tool", tool=tc.name, result=result)

            if result.ok:
                same_failures = 0
                last_fp = None
                user_denies_consecutive = 0
            else:
                if result.error == "UserDenied":
                    user_denies_total += 1
                    user_denies_consecutive += 1
                    if tc.name in user_denies_by_action:
                        user_denies_by_action[tc.name] += 1
                    if _too_many_user_denials(user_denies_total, user_denies_consecutive, user_denies_by_action, tc.name):
                        return _report(session, False, "用户多次拒绝高风险动作，任务停止。", "user_denied", start_turns, step)
                if result.error in SAFETY_ERRORS:
                    safety += 1
                    if safety >= MAX_SAFETY_REJECTIONS:
                        return _report(session, False, "多次越权或危险命令", "safety_rejections", start_turns, step)
                fp = _fingerprint(tc.name, tc.arguments)
                same_failures = same_failures + 1 if fp == last_fp else 1
                last_fp = fp
                if same_failures >= MAX_SAME_TOOL_FAILURES:
                    return _report(session, False, "连续同类工具失败", "same_tool_failures", start_turns, step)

    return _report(session, False, "达到最大步数", "max_steps", start_turns, step)


# ---- 内部 ----

def _execute_action(session, action, args, task_grants):
    """裁决 + 授权 + 执行：allow 直接跑，deny 回填 PolicyDenied，ask 支持 once/remember/deny。"""
    decision = decide(session.mode, action)
    if decision.verdict == "deny":
        return ToolResult(ok=False, output=decision.reason, error="PolicyDenied")
    if decision.verdict == "ask":
        key = grant_key(action, args)
        if key not in task_grants:
            choice = normalize_ask_choice(session.asker(action, args) if session.asker else ASK_DENY)
            if choice == ASK_DENY:
                return ToolResult(ok=False, output="用户拒绝执行该动作。", error="UserDenied")
            if choice == ASK_REMEMBER:
                task_grants.add(key)
            elif choice != ASK_ONCE:
                return ToolResult(ok=False, output="用户拒绝执行该动作。", error="UserDenied")
    return session.run_tool_fn(action, args, session.workspace)


def _too_many_user_denials(total, consecutive, by_action, action):
    if consecutive >= MAX_CONSECUTIVE_USER_DENIALS:
        return True
    if total >= MAX_USER_DENIALS:
        return True
    return by_action.get(action, 0) >= MAX_USER_DENIALS_BY_ACTION


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
    successful_test_turns = [t for t in test_turns if t.result.ok]
    result_turn = (successful_test_turns or test_turns)[-1] if test_turns else None
    test_result = _clip(result_turn.result.output, 2000) if result_turn else ""
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
        lines.append("过程中遇到的问题：")
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
