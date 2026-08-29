"""工具层：一个接口 + 一个结果类型 + 注册表。

安全约束（路径 / 命令策略 / 凭据）作为 helper 内聚在本文件。模型只能通过
run_tool 调用白名单工具；预期内失败统一返回 ToolResult(ok=False)，供 agent
loop 作为 observation 回填给模型。
"""

import fnmatch
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# 工具级常量（测试通过 monkeypatch 覆盖这些值）
MAX_READ_BYTES = 100 * 1024
COMMAND_TIMEOUT = 30
MAX_COMMAND_OUTPUT = 12_000


class ToolError(Exception):
    """预期内的工具失败，由 run_tool 统一转成 ToolResult(ok=False)。"""

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    error: str = ""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    execute: Callable[[dict, Path], ToolResult]


SENSITIVE_FILENAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}

SENSITIVE_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}

DENIED_COMMAND_WORDS = {
    "rm",
    "del",
    "erase",
    "rmdir",
    "rd",
    "remove-item",
    "pip",
    "npm",
    "pnpm",
    "yarn",
    "curl",
    "wget",
    "ssh",
    "scp",
}

# 前缀白名单：命中即 allow，比黑名单更具体（`npm test` 允许，`npm install` 仍拒）。
ALLOWED_COMMAND_PREFIXES = (
    "python",
    "pytest",
    "npm test",
    "cargo test",
    "go test",
)

SHELL_METACHARS = ("&&", "||", ";", "|", ">", "<", "`", "$(")  # 引号外的操作符；\n/\r 在 check_command_policy 单独处理


def run_tool(name: str, args: dict, workspace: Path) -> ToolResult:
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(ok=False, output="", error="UnknownTool")
    if not isinstance(args, dict):
        return ToolResult(ok=False, output="", error="InvalidArgs")

    try:
        return tool.execute(args, workspace)
    except ToolError as exc:
        return ToolResult(ok=False, output=exc.message, error=exc.error_type)
    except Exception as exc:  # pragma: no cover - defensive boundary for the loop.
        return ToolResult(ok=False, output=str(exc), error=exc.__class__.__name__)


def resolve_in_workspace(path: str, workspace: Path) -> Path:
    if not isinstance(path, str) or path == "":
        raise ToolError("InvalidArgs", "path 必须是非空字符串")

    root = workspace.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ToolError("PathOutsideWorkspace", f"path outside workspace: {path}") from exc
    return resolved


def check_command_policy(command: str) -> str:
    if not isinstance(command, str) or command.strip() == "":
        return "deny"

    # 换行始终拒绝（多行命令注入）
    if "\n" in command or "\r" in command:
        return "deny"

    # 引号外的 shell 操作符才算元字符；引号内是字符串内容
    if _has_unquoted_metachar(command):
        return "deny"

    try:
        parts = shlex.split(command, posix=True)
    except ValueError:
        return "deny"
    if not parts:
        return "deny"

    normalized = " ".join(parts).lower()

    # 前缀白名单优先于黑名单：否则 `npm test` 会因 `npm` 在黑名单而被误拒。
    if _matches_allow_prefix(normalized):
        return "allow"

    first = _command_basename(parts[0]).lower()
    if first in DENIED_COMMAND_WORDS:
        return "deny"

    return "deny"


def _matches_allow_prefix(normalized: str) -> bool:
    for prefix in ALLOWED_COMMAND_PREFIXES:
        if normalized == prefix or normalized.startswith(prefix + " "):
            return True
    return False


def _has_unquoted_metachar(command: str) -> bool:
    """检查 shell 操作符是否出现在引号外（引号内是字符串内容，不算元字符）。"""
    in_quote = None
    escaped = False
    i = 0
    while i < len(command):
        ch = command[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if in_quote:
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            in_quote = ch
            i += 1
            continue
        for meta in SHELL_METACHARS:
            if command.startswith(meta, i):
                return True
        i += 1
    return False


def _command_basename(value: str) -> str:
    return Path(value.strip('"')).name


def _require_string(args: dict, key: str, default: str | None = None) -> str:
    value = args.get(key, default)
    if not isinstance(value, str) or value == "":
        raise ToolError("InvalidArgs", f"{key} 必须是非空字符串")
    return value


def _ensure_not_sensitive(path: Path) -> None:
    name = path.name.lower()
    if name in SENSITIVE_FILENAMES or any(name.endswith(s) for s in SENSITIVE_SUFFIXES):
        raise ToolError("PolicyDenied", f"拒绝访问凭据文件: {path.name}")


def _relative(path: Path, workspace: Path) -> str:
    return path.resolve().relative_to(workspace.resolve()).as_posix()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, limit)
    return text[:keep] + f"\n...[截断，原始输出 {len(text)} 字符]"


def _list_files(args: dict, workspace: Path) -> ToolResult:
    target = resolve_in_workspace(_require_string(args, "path", "."), workspace)
    if not target.exists():
        raise ToolError("NotFound", f"path not found: {args.get('path', '.')}")
    if not target.is_dir():
        raise ToolError("InvalidArgs", "list_files path 必须是目录")

    entries = []
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        suffix = "/" if child.is_dir() else ""
        entries.append(_relative(child, workspace) + suffix)
    return ToolResult(ok=True, output="\n".join(entries) if entries else "空目录")


def _read_file(args: dict, workspace: Path) -> ToolResult:
    target = resolve_in_workspace(_require_string(args, "path"), workspace)
    _ensure_not_sensitive(target)
    if not target.exists():
        raise ToolError("FileNotFound", f"file not found: {args.get('path')}")
    if not target.is_file():
        raise ToolError("InvalidArgs", "read_file path 必须是文件")

    data = target.read_bytes()
    truncated = len(data) > MAX_READ_BYTES
    if truncated:
        data = data[:MAX_READ_BYTES]
    text = data.decode("utf-8", errors="replace")
    if truncated:
        text += f"\n...[截断，原始文件 {target.stat().st_size} 字节]"
    return ToolResult(ok=True, output=text)


def _write_file(args: dict, workspace: Path) -> ToolResult:
    target = resolve_in_workspace(_require_string(args, "path"), workspace)
    _ensure_not_sensitive(target)
    content = args.get("content")
    if not isinstance(content, str):
        raise ToolError("InvalidArgs", "content 必须是字符串")

    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    target.write_text(content, encoding="utf-8")
    verb = "updated" if existed else "created"
    return ToolResult(ok=True, output=f"{verb}: {_relative(target, workspace)}")


def _glob(args: dict, workspace: Path) -> ToolResult:
    pattern = _require_string(args, "pattern")
    matches = []
    for path in workspace.resolve().glob(pattern):
        if path.is_file() or path.is_dir():
            if _is_sensitive_path(path):
                continue
            suffix = "/" if path.is_dir() else ""
            matches.append(_relative(path, workspace) + suffix)
    matches.sort()
    return ToolResult(ok=True, output="\n".join(matches) if matches else "无匹配")


def _grep(args: dict, workspace: Path) -> ToolResult:
    pattern = _require_string(args, "pattern")
    base = resolve_in_workspace(_require_string(args, "path", "."), workspace)
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ToolError("InvalidArgs", f"invalid regex: {exc}") from exc
    if not base.exists():
        raise ToolError("NotFound", f"path not found: {args.get('path', '.')}")

    files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
    lines = []
    for file in files:
        if _is_sensitive_path(file):
            continue
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                lines.append(f"{_relative(file, workspace)}:{lineno}: {line}")
    return ToolResult(ok=True, output="\n".join(lines) if lines else "无匹配")


def _is_sensitive_path(path: Path) -> bool:
    name = path.name.lower()
    return name in SENSITIVE_FILENAMES or any(name.endswith(s) for s in SENSITIVE_SUFFIXES)


def _run_command(args: dict, workspace: Path) -> ToolResult:
    command = _require_string(args, "command")
    if check_command_policy(command) != "allow":
        raise ToolError("PolicyDenied", f"command denied: {command}")

    try:
        parts = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ToolError("InvalidArgs", f"invalid command: {exc}") from exc

    # 让 `python ...` 用 agent 自己的解释器（含 pytest），避免 PATH 解析到无 pytest 的 Python
    if parts and parts[0].lower() in ("python", "python.exe"):
        parts[0] = sys.executable

    try:
        completed = subprocess.run(
            parts,
            cwd=workspace.resolve(),
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        return ToolResult(ok=False, output=_truncate(output, MAX_COMMAND_OUTPUT), error="Timeout")
    except OSError as exc:
        return ToolResult(ok=False, output=str(exc), error="CommandError")

    output = completed.stdout
    if completed.stderr:
        output += ("\n" if output else "") + completed.stderr
    output = _truncate(output, MAX_COMMAND_OUTPUT)
    if completed.returncode != 0:
        return ToolResult(ok=False, output=output, error="CommandFailed")
    return ToolResult(ok=True, output=output)


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


TOOLS: dict[str, Tool] = {
    "list_files": Tool(
        name="list_files",
        description="列出 workspace 内指定目录的直接子项。",
        parameters=_schema({"path": {"type": "string", "default": "."}}),
        execute=_list_files,
    ),
    "read_file": Tool(
        name="read_file",
        description="读取 workspace 内的 UTF-8 文本文件，超长内容会截断。",
        parameters=_schema({"path": {"type": "string"}}, ["path"]),
        execute=_read_file,
    ),
    "glob": Tool(
        name="glob",
        description="使用 glob pattern 查找 workspace 内的文件或目录。",
        parameters=_schema({"pattern": {"type": "string"}}, ["pattern"]),
        execute=_glob,
    ),
    "grep": Tool(
        name="grep",
        description="在 workspace 内按正则表达式搜索文本内容。",
        parameters=_schema(
            {"pattern": {"type": "string"}, "path": {"type": "string", "default": "."}},
            ["pattern"],
        ),
        execute=_grep,
    ),
    "write_file": Tool(
        name="write_file",
        description="覆盖写入 workspace 内的文本文件，禁止写入凭据文件。",
        parameters=_schema({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        execute=_write_file,
    ),
    "run_command": Tool(
        name="run_command",
        description="在 workspace 内执行 allowlist 中的本地命令，带超时和输出截断。",
        parameters=_schema({"command": {"type": "string"}}, ["command"]),
        execute=_run_command,
    ),
}
