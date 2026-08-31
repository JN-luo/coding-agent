"""REPL 外壳：维护一个长期 session。"""

import argparse
import sys
from pathlib import Path

from agent.config import ConfigError, load_config
from agent.llm import LLM, LLMError
from agent.loop import MAX_STEPS, render_report
from agent.prompts import build_system_prompt
from agent.session import Session
from agent.tools import TOOLS
from agent.trace import start_trace


def _short(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def _flat(value: str) -> str:
    """把换行/多余空白压成单空格，避免 args 里的 content 拆散 action 行。"""
    return " ".join(value.split())


def _print_step(kind: str, **fields) -> None:
    if kind == "action":
        args = fields.get("args", {})
        args_str = ", ".join(f"{k}={_short(_flat(str(v)), 40)}" for k, v in args.items())
        label = f"{fields['action']}({args_str})" if args_str else fields["action"]
        print(f"  [{fields['step']}] -> {label}", flush=True)
    elif kind == "parse_error":
        print(f"  [{fields['step']}] parse_error: 模型输出不是合法动作，已要求重试", flush=True)
    elif kind == "tool":
        result = fields["result"]
        tool = fields.get("tool", "")
        mark = "ok" if result.ok else f"FAIL {result.error}"
        print(f"    {mark}", flush=True)
        if tool == "read_file":
            print(f"    （{len(result.output.splitlines())} 行）", flush=True)
        else:
            out = result.output.strip()
            if out:
                print("    " + _short(out, 200).replace("\n", "\n    "), flush=True)


def _ask(action: str, args: dict) -> bool:
    args_str = ", ".join(f"{k}={_short(_flat(str(v)), 40)}" for k, v in args.items())
    label = f"{action}({args_str})" if args_str else action
    answer = input(f"  允许本任务内继续执行同类动作 {label} 吗？[y/N] ").strip().lower()
    return answer in ("y", "yes")


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        cfg = load_config()
        llm = LLM(cfg)
    except (ConfigError, LLMError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 1

    workspace = Path(args.workspace).resolve()
    with start_trace() as tracer:
        mode = "readonly" if args.readonly else ("auto" if args.auto else "ask")
        session = Session(
            llm,
            workspace,
            trace=tracer,
            on_step=_print_step,
            mode=mode,
            asker=_ask,
            max_steps=args.max_steps,
        )
        session.set_system(build_system_prompt(TOOLS))

        if args.task:
            print(render_report(session.submit(args.task)))
        else:
            _repl(session)

    return 0


def _repl(session: Session) -> None:
    print("Coding Agent REPL（空行或 exit / quit 退出）")
    while True:
        try:
            task = input("❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if task in ("", "exit", "quit"):
            break
        print(render_report(session.submit(task)))
        print()


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="python -m agent", description="本地 coding agent")
    p.add_argument("task", nargs="?", help="单次任务；省略则进入 REPL")
    p.add_argument("--workspace", default=".", help="工作目录，默认当前目录")
    p.add_argument("--readonly", action="store_true", help="只读模式：禁止 write_file / run_command")
    p.add_argument("--auto", action="store_true", help="自动模式：写文件与命令直接执行")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS, help=f"单任务最大步数，默认 {MAX_STEPS}")
    return p.parse_args(argv)
