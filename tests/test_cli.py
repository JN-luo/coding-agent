"""cli.py 的单元测试。

覆盖 DESIGN.md §3 cli.py：
  - _parse_args：task / --workspace（默认当前目录）
  - _repl：空行 / exit / quit 退出，其余提交任务
  - main：one-shot 调 session.submit；无 task 进 REPL；配置错误返回 1
"""

import builtins
from contextlib import contextmanager

import agent.cli as cli
from agent.config import ConfigError


class _FakeSession:
    def __init__(self):
        self.submitted = []
        self.system = None

    def set_system(self, rules):
        self.system = rules

    def submit(self, task):
        self.submitted.append(task)
        return "REPORT"


def _fake_inputs(values):
    it = iter(values)

    def fake(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    return fake


# ---- _parse_args ----

def test_parse_args_task_and_workspace():
    args = cli._parse_args(["任务", "--workspace", "x"])
    assert args.task == "任务"
    assert args.workspace == "x"


def test_parse_args_default_workspace():
    args = cli._parse_args(["任务"])
    assert args.task == "任务"
    assert args.workspace == "."


# ---- _repl ----

def test_repl_exits_on_empty(monkeypatch):
    s = _FakeSession()
    monkeypatch.setattr(builtins, "input", _fake_inputs([""]))
    cli._repl(s)
    assert s.submitted == []


def test_repl_exits_on_exit_and_quit(monkeypatch):
    for word in ("exit", "quit"):
        s = _FakeSession()
        monkeypatch.setattr(builtins, "input", _fake_inputs([word]))
        cli._repl(s)
        assert s.submitted == []


def test_repl_submits_tasks(monkeypatch):
    s = _FakeSession()
    monkeypatch.setattr(builtins, "input", _fake_inputs(["任务1", "任务2", ""]))
    monkeypatch.setattr(cli, "render_report", lambda r: r)
    cli._repl(s)
    assert s.submitted == ["任务1", "任务2"]


# ---- main ----

def test_main_config_error_returns_1(monkeypatch, capsys):
    def boom():
        raise ConfigError("bad temperature")

    monkeypatch.setattr(cli, "load_config", boom)
    assert cli.main(["任务"]) == 1
    assert "bad temperature" in capsys.readouterr().err


def test_main_one_shot(monkeypatch):
    s = _FakeSession()

    @contextmanager
    def fake_trace():
        yield object()

    monkeypatch.setattr(cli, "load_config", lambda: object())
    monkeypatch.setattr(cli, "LLM", lambda cfg: object())
    monkeypatch.setattr(cli, "Session", lambda llm, ws, trace=None: s)
    monkeypatch.setattr(cli, "start_trace", fake_trace)
    monkeypatch.setattr(cli, "build_system_prompt", lambda tools: "rules")
    monkeypatch.setattr(cli, "render_report", lambda r: r)
    assert cli.main(["任务"]) == 0
    assert s.submitted == ["任务"]


def test_main_no_task_enters_repl(monkeypatch):
    s = _FakeSession()

    @contextmanager
    def fake_trace():
        yield object()

    monkeypatch.setattr(cli, "load_config", lambda: object())
    monkeypatch.setattr(cli, "LLM", lambda cfg: object())
    monkeypatch.setattr(cli, "Session", lambda llm, ws, trace=None: s)
    monkeypatch.setattr(cli, "start_trace", fake_trace)
    monkeypatch.setattr(cli, "build_system_prompt", lambda tools: "rules")
    monkeypatch.setattr(builtins, "input", _fake_inputs([""]))  # 空行退出 REPL
    monkeypatch.setattr(cli, "render_report", lambda r: r)
    assert cli.main([]) == 0
    assert s.submitted == []
