"""trace.py 的单元测试。

覆盖 DESIGN.md §3 trace.py：JSONL 落盘、逐行 flush、按 run 命名、stderr 打路径。
"""

import json

from agent.trace import Tracer, start_trace


def test_tracer_log_writes_jsonl(tmp_path):
    p = tmp_path / "t.jsonl"
    with Tracer(p) as t:
        t.log("tool", tool="read_file", ok=True)
        t.log("stop", reason="max_steps", steps=20)

    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "tool"
    assert first["tool"] == "read_file"
    assert first["ok"] is True
    assert isinstance(first["ts"], (int, float))
    assert json.loads(lines[1])["event"] == "stop"


def test_tracer_flushes_each_line(tmp_path):
    p = tmp_path / "t.jsonl"
    with Tracer(p) as t:
        t.log("action", action="list_files")
    assert "list_files" in p.read_text(encoding="utf-8")  # 未 close 也能读到


def test_tracer_keeps_chinese(tmp_path):
    p = tmp_path / "t.jsonl"
    with Tracer(p) as t:
        t.log("parse_error", err="自然语言兜底失败")
    assert "自然语言兜底失败" in p.read_text(encoding="utf-8")


def test_start_trace_creates_run_file(tmp_path, capsys):
    with start_trace(tmp_path) as t:
        t.log("tool", tool="read_file")

    files = list(tmp_path.glob("run-*.jsonl"))
    assert len(files) == 1
    err = capsys.readouterr().err
    assert "trace:" in err
    assert str(files[0]) in err


def test_start_trace_respects_env_dir(tmp_path, monkeypatch):
    target = tmp_path / "logs"
    monkeypatch.setenv("CODING_AGENT_TRACE_DIR", str(target))
    with start_trace() as t:
        pass
    assert target.exists()
    assert list(target.glob("run-*.jsonl"))
