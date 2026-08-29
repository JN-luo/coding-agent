"""llm 真实接入 smoke 测试（手动跑，不进 pytest）。

用法（在仓库根目录运行，确保能读到 .env）：
    python scripts/smoke_llm.py

前置：在 .env 或环境变量里配好 OPENAI_API_KEY / OPENAI_BASE_URL /
CODING_AGENT_MODEL / CODING_AGENT_TEMPERATURE（见 DESIGN.md §3）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import load_config
from agent.llm import LLM, LLMError
from agent.messages import Turn
from agent.parser import ParseError, parse
from agent.prompts import build_system_prompt
from agent.tools import TOOLS, ToolResult


def main() -> int:
    print("== 1. 加载配置 ==")
    try:
        cfg = load_config()
        llm = LLM(cfg)
    except LLMError as exc:
        print(f"[FAIL] {exc}")
        return 1
    print(f"[OK] model={cfg.model!r} base_url={cfg.base_url!r} temperature={cfg.temperature}")

    print("\n== 2. complete（主循环接口）==")
    messages = [
        {"role": "system", "content": build_system_prompt(TOOLS)},
        {"role": "user", "content": "列出当前目录下的文件。"},
    ]
    try:
        raw = llm.complete(messages)
    except LLMError as exc:
        print(f"[FAIL] complete 调用失败: {exc}")
        return 1
    print(f"raw = {raw}")
    try:
        action = parse(raw, frozenset(TOOLS))
        print(f"[OK] 解析成功: action={action.action} args={action.args}")
    except ParseError as exc:
        print(f"[WARN] 输出可解析性失败（不一定是接入问题，可能是模型跑偏）: {exc}")

    print("\n== 3. summarize（语义压缩接口）==")
    turns = [
        Turn("read_file", {"path": "pyproject.toml"}, ToolResult(ok=True, output="[tool.pytest]")),
        Turn("run_command", {"command": "python -m pytest"}, ToolResult(ok=False, output="", error="CommandFailed")),
    ]
    try:
        summary = llm.summarize(turns)
    except LLMError as exc:
        print(f"[FAIL] summarize 调用失败: {exc}")
        return 1
    print(f"summary =\n{summary}")
    print("[OK] summarize 返回非空" if summary.strip() else "[WARN] summarize 返回空串")

    print("\n== smoke 完成 ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
