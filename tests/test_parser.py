"""parser.py 的单元测试。

覆盖 DESIGN.md §3 parser.py 定义的解析契约，按边界分块：
  1) 正常 JSON 动作 / final 动作
  2) JSON 外包裹 Markdown 代码块
  3) JSON 缺字段
  4) 未知工具名
  5) 参数类型错误（args 必须是对象）
  6) 纯自然语言 / 非 JSON 兜底报错

每个测试名对应一条边界，答辩时可直接引用。
"""

import pytest

from agent.parser import Action, ParseError, parse

# 代表性工具集合，仅用于隔离测试 parser；真实运行时 loop 会传 frozenset(TOOLS)。
VALID_TOOLS = frozenset(
    {"list_files", "read_file", "glob", "grep", "write_file", "run_command"}
)


# ---- 1) 正常解析 ----

def test_parses_plain_tool_action():
    action = parse(
        '{"thought": "先看结构", "action": "list_files", "args": {"path": "."}}',
        VALID_TOOLS,
    )
    assert action == Action(action="list_files", args={"path": "."}, thought="先看结构")


def test_parses_final_action():
    action = parse('{"action": "final", "message": "已完成并测试通过"}', VALID_TOOLS)
    assert action.action == "final"
    assert action.message == "已完成并测试通过"


def test_final_bypasses_tool_whitelist():
    # "final" 不是工具名，不应被白名单拦截
    action = parse('{"action": "final", "message": "ok"}', VALID_TOOLS)
    assert action.action == "final"


def test_thought_is_optional():
    action = parse('{"action": "read_file", "args": {"path": "a.py"}}', VALID_TOOLS)
    assert action.thought == ""


def test_args_optional_defaults_to_empty():
    action = parse('{"action": "list_files"}', VALID_TOOLS)
    assert action.args == {}


# ---- 2) Markdown 代码块 ----

def test_strips_json_fence():
    raw = '```json\n{"action": "list_files", "args": {"path": "."}}\n```'
    action = parse(raw, VALID_TOOLS)
    assert action.action == "list_files"
    assert action.args == {"path": "."}


def test_strips_fence_without_language():
    raw = '```\n{"action": "list_files", "args": {}}\n```'
    action = parse(raw, VALID_TOOLS)
    assert action.action == "list_files"


def test_extracts_json_surrounded_by_prose():
    # 真实模型常在 JSON 前后夹带自然语言
    raw = '好的，我先查看目录：\n{"action": "list_files", "args": {"path": "."}}\n然后继续。'
    action = parse(raw, VALID_TOOLS)
    assert action.action == "list_files"


# ---- 3) JSON 缺字段 ----

def test_missing_action_field_raises():
    with pytest.raises(ParseError):
        parse('{"args": {"path": "."}}', VALID_TOOLS)


def test_empty_object_raises():
    with pytest.raises(ParseError):
        parse('{}', VALID_TOOLS)


def test_final_requires_message():
    with pytest.raises(ParseError):
        parse('{"action": "final"}', VALID_TOOLS)


def test_final_rejects_empty_message():
    with pytest.raises(ParseError):
        parse('{"action": "final", "message": ""}', VALID_TOOLS)


# ---- 4) 未知工具名 ----

def test_unknown_tool_raises():
    with pytest.raises(ParseError):
        parse('{"action": "delete_everything", "args": {}}', VALID_TOOLS)


# ---- 5) 参数类型错误 ----

def test_args_must_be_object__string():
    with pytest.raises(ParseError):
        parse('{"action": "read_file", "args": "not_a_dict"}', VALID_TOOLS)


def test_args_must_be_object__list():
    with pytest.raises(ParseError):
        parse('{"action": "read_file", "args": [1, 2]}', VALID_TOOLS)


def test_args_null_treated_as_empty():
    # 真实模型可能输出 "args": null，宽松处理为 {}
    action = parse('{"action": "list_files", "args": null}', VALID_TOOLS)
    assert action.args == {}


# ---- 6) 自然语言 / 非 JSON 兜底 ----

def test_non_json_prose_raises():
    with pytest.raises(ParseError):
        parse('我觉得应该先看看项目结构', VALID_TOOLS)


def test_json_array_not_object_raises():
    with pytest.raises(ParseError):
        parse('[1, 2, 3]', VALID_TOOLS)


def test_whitespace_only_raises():
    with pytest.raises(ParseError):
        parse('   ', VALID_TOOLS)
