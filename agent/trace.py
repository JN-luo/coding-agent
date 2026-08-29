"""调试轨迹（JSONL 落盘，仅日志，不参与控制流）。

append-only，每行一个事件、逐行 flush，崩溃时尽量保住已写行。
只做观察，不做 replay / resume / 恢复。

事件名（见 DESIGN.md §3）：task / llm_raw / action / parse_error / tool / stop。
"""

import json
import os
import sys
import time
from pathlib import Path

DEFAULT_TRACE_DIR = "trace"


class NullTracer:
    """不落盘的 no-op tracer（无 trace 场景 / 测试）。"""

    def log(self, event, **fields):
        pass


class Tracer:
    """把事件追加到 JSONL 文件，每行 flush。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = path.open("a", encoding="utf-8")

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def log(self, event: str, **fields) -> None:
        line = json.dumps({"ts": time.time(), "event": event, **fields}, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def start_trace(trace_dir: str | Path | None = None) -> Tracer:
    """创建按 run 命名的轨迹文件，往 stderr 打一行路径，返回 Tracer。

    目录取自 `CODING_AGENT_TRACE_DIR` 环境变量，缺省为 DEFAULT_TRACE_DIR。
    """
    base = Path(trace_dir or os.environ.get("CODING_AGENT_TRACE_DIR", DEFAULT_TRACE_DIR))
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    print(f"trace: {path}", file=sys.stderr)
    return Tracer(path)
