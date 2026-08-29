"""prompts.py 的单元测试。

只测「由代码生成」的结构：每个工具名 + description 都进 prompt，工具列表行派生。
不测手写 prose 字面量——prompt 有效性靠 demo 预演（DESIGN.md §9）兜底。
"""

from agent.prompts import build_system_prompt
from agent.tools import TOOLS


def test_build_system_prompt_derives_all_tools():
    prompt = build_system_prompt(TOOLS)
    for name, tool in TOOLS.items():
        assert name in prompt
        assert tool.description in prompt
    assert "可用工具：" in prompt
    assert "、final" in prompt
