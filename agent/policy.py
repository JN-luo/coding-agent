"""权限层：位于 parser 和 tools 之间（DESIGN.md §3）。

LLM 只负责提出动作，runtime 根据任务模式裁决 allow / ask / deny。
安全边界分层：prompt 负责引导，policy 负责模式权限，tools 负责底层沙箱。
"""

from dataclasses import dataclass

READ_TOOLS = frozenset({"list_files", "read_file", "glob", "grep"})


@dataclass(frozen=True)
class Decision:
    verdict: str   # "allow" | "ask" | "deny"
    reason: str = ""


def decide(mode: str, action: str) -> Decision:
    """按模式裁决一个动作。读工具始终 allow。"""
    if action in READ_TOOLS:
        return Decision("allow")
    if mode == "auto":
        return Decision("allow")
    if mode == "readonly":
        return Decision("deny", "只读模式，禁止 write_file / run_command")
    return Decision("ask")  # 默认 ask


def grant_key(action: str, args: dict) -> str:
    """授权粒度：write_file 按 action；run_command 按完整命令。"""
    if action == "run_command":
        return f"run_command:{args.get('command', '')}"
    return action
