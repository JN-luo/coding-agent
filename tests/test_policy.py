"""policy.py 的单元测试。

覆盖 DESIGN.md §3 policy.py：三模式裁决（ask / readonly / auto）与授权粒度。
"""

from agent.policy import decide, grant_key


def test_read_tools_always_allow():
    for tool in ("list_files", "read_file", "glob", "grep"):
        assert decide("ask", tool).verdict == "allow"
        assert decide("readonly", tool).verdict == "allow"
        assert decide("auto", tool).verdict == "allow"


def test_write_ask_by_default():
    assert decide("ask", "write_file").verdict == "ask"
    assert decide("ask", "run_command").verdict == "ask"


def test_write_deny_in_readonly():
    d = decide("readonly", "write_file")
    assert d.verdict == "deny"
    assert d.reason


def test_write_allow_in_auto():
    assert decide("auto", "write_file").verdict == "allow"
    assert decide("auto", "run_command").verdict == "allow"


def test_grant_key_granularity():
    assert grant_key("write_file", {"path": "a.py"}) == "write_file"
    assert grant_key("run_command", {"command": "pytest -q"}) == "run_command:pytest -q"
    assert grant_key("run_command", {"command": "pytest -q test.py"}) == "run_command:pytest -q test.py"
