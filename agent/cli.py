"""REPL 外壳：维护一个长期 session。"""

import argparse
import sys
from pathlib import Path

from agent.config import ConfigError, load_config
from agent.llm import LLM, LLMError
from agent.loop import render_report
from agent.prompts import build_system_prompt
from agent.session import Session
from agent.tools import TOOLS
from agent.trace import start_trace


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
        session = Session(llm, workspace, trace=tracer)
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
    return p.parse_args(argv)
