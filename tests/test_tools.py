"""tools.py 的单元测试。

覆盖 DESIGN.md §3 tools.py：
  - 注册表（6 个工具，每个有 description + JSON Schema）
  - 统一 ToolResult / run_tool 分发
  - 路径安全（resolve_in_workspace，越权拒绝）
  - 凭据保护（.env / 私钥）
  - 命令策略（allow / deny / 元字符）
  - 各工具行为（读 / 写 / 列 / glob / grep / run_command）
  - 超时与截断（monkeypatch 常量）

文件类工具用 pytest 内置的 tmp_path 作为 workspace。
"""

import pytest

from agent import tools as tools_mod
from agent.tools import (
    TOOLS,
    ToolError,
    check_command_policy,
    resolve_in_workspace,
    run_tool,
)


# ---- 注册表 ----

def test_registry_has_six_tools():
    assert frozenset(TOOLS) == {
        "list_files",
        "read_file",
        "glob",
        "grep",
        "write_file",
        "run_command",
    }


def test_each_tool_has_schema():
    for tool in TOOLS.values():
        assert tool.description, f"{tool.name} 缺 description"
        assert isinstance(tool.parameters, dict), f"{tool.name} 缺 parameters"


def test_run_unknown_tool(tmp_path):
    result = run_tool("nonexistent", {}, tmp_path)
    assert result.ok is False
    assert result.error == "UnknownTool"


# ---- 路径安全 ----

def test_resolve_inside_workspace(tmp_path):
    assert resolve_in_workspace("a.py", tmp_path) == (tmp_path / "a.py").resolve()


def test_resolve_dot_returns_workspace(tmp_path):
    assert resolve_in_workspace(".", tmp_path) == tmp_path.resolve()


def test_resolve_dotdot_rejected(tmp_path):
    with pytest.raises(ToolError) as e:
        resolve_in_workspace("../outside.txt", tmp_path)
    assert e.value.error_type == "PathOutsideWorkspace"


def test_resolve_absolute_outside_rejected(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    with pytest.raises(ToolError) as e:
        resolve_in_workspace(str(outside), tmp_path)
    assert e.value.error_type == "PathOutsideWorkspace"


# ---- 凭据保护 ----

def test_read_env_rejected(tmp_path):
    (tmp_path / ".env").write_text("SECRET=1")
    result = run_tool("read_file", {"path": ".env"}, tmp_path)
    assert result.ok is False
    assert result.error == "PolicyDenied"


def test_write_env_rejected(tmp_path):
    result = run_tool("write_file", {"path": ".env", "content": "x"}, tmp_path)
    assert result.ok is False
    assert result.error == "PolicyDenied"
    assert not (tmp_path / ".env").exists()


def test_read_private_key_rejected(tmp_path):
    (tmp_path / "id_rsa").write_text("PRIVATE KEY")
    result = run_tool("read_file", {"path": "id_rsa"}, tmp_path)
    assert result.ok is False
    assert result.error == "PolicyDenied"


def test_write_pem_rejected(tmp_path):
    result = run_tool("write_file", {"path": "server.pem", "content": "x"}, tmp_path)
    assert result.ok is False
    assert result.error == "PolicyDenied"
    assert not (tmp_path / "server.pem").exists()


def test_grep_skips_credentials(tmp_path):
    (tmp_path / ".env").write_text("SECRET=1\n")
    (tmp_path / "a.py").write_text("SECRET=1\n")
    result = run_tool("grep", {"pattern": "SECRET", "path": "."}, tmp_path)
    assert result.ok is True
    assert "a.py" in result.output
    assert ".env" not in result.output


# ---- read_file ----

def test_read_file_ok(tmp_path):
    (tmp_path / "a.py").write_text("print('hi')")
    result = run_tool("read_file", {"path": "a.py"}, tmp_path)
    assert result.ok is True
    assert "print('hi')" in result.output


def test_read_file_missing(tmp_path):
    result = run_tool("read_file", {"path": "nope.py"}, tmp_path)
    assert result.ok is False
    assert result.error == "FileNotFound"


def test_read_file_truncates(tmp_path, monkeypatch):
    (tmp_path / "big.txt").write_text("x" * 100)
    monkeypatch.setattr(tools_mod, "MAX_READ_BYTES", 10)
    result = run_tool("read_file", {"path": "big.txt"}, tmp_path)
    assert result.ok is True
    assert len(result.output) < 100
    assert "截断" in result.output


# ---- write_file ----

def test_write_file_create(tmp_path):
    result = run_tool("write_file", {"path": "new.py", "content": "x = 1"}, tmp_path)
    assert result.ok is True
    assert (tmp_path / "new.py").read_text() == "x = 1"


def test_write_file_overwrite(tmp_path):
    (tmp_path / "a.py").write_text("old")
    result = run_tool("write_file", {"path": "a.py", "content": "new"}, tmp_path)
    assert result.ok is True
    assert (tmp_path / "a.py").read_text() == "new"


def test_write_file_outside_rejected(tmp_path):
    result = run_tool("write_file", {"path": "../x.py", "content": "x"}, tmp_path)
    assert result.ok is False
    assert result.error == "PathOutsideWorkspace"


def test_write_file_empty_content(tmp_path):
    result = run_tool("write_file", {"path": "empty.txt", "content": ""}, tmp_path)
    assert result.ok is True
    assert (tmp_path / "empty.txt").read_text() == ""


# ---- list_files ----

def test_list_files_ok(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "sub").mkdir()
    result = run_tool("list_files", {"path": "."}, tmp_path)
    assert result.ok is True
    assert "a.py" in result.output
    assert "sub" in result.output


def test_list_files_missing(tmp_path):
    result = run_tool("list_files", {"path": "nope"}, tmp_path)
    assert result.ok is False
    assert result.error == "NotFound"


# ---- glob ----

def test_glob_recursive(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("x")
    result = run_tool("glob", {"pattern": "**/*.py"}, tmp_path)
    assert result.ok is True
    assert "a.py" in result.output
    assert "b.py" in result.output


def test_glob_no_match(tmp_path):
    result = run_tool("glob", {"pattern": "*.txt"}, tmp_path)
    assert result.ok is True
    assert "无匹配" in result.output


# ---- grep ----

def test_grep_finds(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    result = run_tool("grep", {"pattern": "foo", "path": "."}, tmp_path)
    assert result.ok is True
    assert "a.py" in result.output


def test_grep_no_match(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    result = run_tool("grep", {"pattern": "zzz", "path": "."}, tmp_path)
    assert result.ok is True
    assert "无匹配" in result.output


def test_grep_invalid_regex(tmp_path):
    result = run_tool("grep", {"pattern": "[", "path": "."}, tmp_path)
    assert result.ok is False
    assert result.error == "InvalidArgs"


# ---- run_command / 命令策略 ----

def test_policy_allows_whitelisted():
    assert check_command_policy("python -m pytest") == "allow"
    assert check_command_policy("python script.py") == "allow"


def test_policy_allows_npm_test():
    # `npm` 在黑名单，但 `npm test` 是更具体的白名单前缀，应优先允许。
    assert check_command_policy("npm test") == "allow"
    assert check_command_policy("npm test --watch") == "allow"


def test_policy_prefix_whitelist():
    assert check_command_policy("cargo test --release") == "allow"
    assert check_command_policy("go test ./...") == "allow"


def test_policy_still_denies_non_whitelisted():
    assert check_command_policy("npm install") == "deny"
    assert check_command_policy("cargo build") == "deny"


def test_policy_denies_denylist():
    assert check_command_policy("rm -rf /") == "deny"
    assert check_command_policy("pip install requests") == "deny"


def test_policy_denies_metachar():
    assert check_command_policy("python -m pytest && rm -rf /") == "deny"


def test_policy_denies_unknown():
    assert check_command_policy("git status") == "deny"


def test_run_command_allowed(tmp_path):
    result = run_tool("run_command", {"command": "python --version"}, tmp_path)
    assert result.ok is True
    assert "Python" in result.output


def test_run_command_denied(tmp_path):
    result = run_tool("run_command", {"command": "rm -rf /"}, tmp_path)
    assert result.ok is False
    assert result.error == "PolicyDenied"


def test_run_command_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_mod, "COMMAND_TIMEOUT", 0.3)
    result = run_tool(
        "run_command", {"command": 'python -c "__import__(\'time\').sleep(3)"'}, tmp_path
    )
    assert result.ok is False
    assert result.error == "Timeout"


def test_run_command_truncates(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_mod, "MAX_COMMAND_OUTPUT", 20)
    result = run_tool("run_command", {"command": "python -c \"print('x'*1000)\""}, tmp_path)
    assert result.ok is True
    assert len(result.output) < 100
    assert "截断" in result.output
