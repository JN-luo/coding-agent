"""LLM 输出解析。

把模型原始输出（可能是裸 JSON、带 Markdown 代码块的 JSON，或夹带自然语言的 JSON）
解析为 Action；任何失败都抛 ParseError，由 loop.py 计数"连续解析失败"。

契约（与 DESIGN.md §3 一致）：
  - parse(raw, valid_tools) -> Action
  - action == "final" 表示最终回答，绕过工具白名单，但必须带非空 message。
  - 工具动作必须命中 valid_tools；args 缺失或为 null 时宽松处理为 {}。
  - 五种边界（代码块 / 缺字段 / 未知工具 / 参数类型 / 自然语言）都在这里处理，
    逐条对应 tests/test_parser.py。
"""

import json
from dataclasses import dataclass, field


class ParseError(Exception):
    """LLM 输出无法解析为合法动作时抛出。"""


@dataclass(frozen=True)
class Action:
    """解析后的一次动作。action == "final" 表示最终回答。"""

    action: str
    args: dict = field(default_factory=dict)
    message: str = ""
    thought: str = ""


def parse(raw: str, valid_tools: frozenset[str]) -> Action:
    """把模型原始输出解析为 Action，失败抛 ParseError。"""
    obj = _load_object(raw)

    action = obj.get("action")
    if not isinstance(action, str) or action == "":
        raise ParseError("缺少非空的 'action' 字段")

    thought = obj.get("thought", "")
    if not isinstance(thought, str):
        thought = ""  # 元数据字段，非字符串时忽略

    if action == "final":
        message = obj.get("message")
        if not isinstance(message, str) or message == "":
            raise ParseError("final 动作必须带非空 'message'")
        return Action(action="final", message=message, thought=thought)

    if action not in valid_tools:
        raise ParseError(f"未知工具名: {action}")

    args = obj.get("args", {})
    if args is None:
        args = {}  # 模型可能输出 "args": null
    if not isinstance(args, dict):
        raise ParseError("'args' 必须是 JSON 对象")

    return Action(action=action, args=args, thought=thought)


def _load_object(raw: str) -> dict:
    """从原始输出中提取并解析出一个 JSON 对象。"""
    text = _extract_json_object(raw)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"不是合法 JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ParseError("JSON 必须是对象，不能是数组或标量")
    return obj


def _extract_json_object(text: str) -> str:
    """定位第一个配平的 `{...}` 对象，容忍代码块与前后自然语言。

    从第一个 `{` 起做一次带字符串感知的括号匹配：代码块围栏（```json / ```）
    不含 `{`，会被自然跳过；字符串内部的 `{`、`}`、转义引号不会干扰匹配。
    """
    start = text.find("{")
    if start == -1:
        raise ParseError("输出中未找到 JSON 对象")

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    raise ParseError("JSON 对象括号未配平")
