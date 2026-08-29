"""配置：只从环境变量或 gitignore 的 .env 读取，凭据不入库。

环境变量（见 DESIGN.md §3 llm.py）：
  - OPENAI_API_KEY            API key（或兼容网关 key）
  - OPENAI_BASE_URL           兼容网关地址
  - CODING_AGENT_MODEL        模型名
  - CODING_AGENT_TEMPERATURE  温度，缺省 0.0

.env 只填充缺失的键，不覆盖已有环境变量（后者优先）。config.local.* 也已由
.gitignore 排除，MVP 只实现 .env。
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TEMPERATURE = 0.0
DOTENV_NAME = ".env"


class ConfigError(Exception):
    """配置值非法。"""


@dataclass(frozen=True)
class Config:
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = DEFAULT_TEMPERATURE


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """从环境变量（可注入 env）与 .env 组装配置。

    注入 env 时不读文件（测试 / 调用方自足）；否则先读 os.environ，再用 .env
    填充缺失键（环境变量优先）。
    """
    if env is not None:
        resolved = dict(env)
    else:
        resolved = dict(os.environ)
        _merge_dotenv(resolved, Path(DOTENV_NAME))

    return Config(
        api_key=resolved.get("OPENAI_API_KEY", ""),
        base_url=resolved.get("OPENAI_BASE_URL", ""),
        model=resolved.get("CODING_AGENT_MODEL", ""),
        temperature=_parse_temperature(resolved.get("CODING_AGENT_TEMPERATURE")),
    )


def _merge_dotenv(env: dict, path: Path) -> None:
    """把 .env 里缺失的键填充进 env，不覆盖已有环境变量。"""
    if not path.is_file():
        return
    for key, value in _parse_dotenv(path):
        env.setdefault(key, value)


def _parse_dotenv(path: Path) -> list[tuple[str, str]]:
    """解析 KEY=VALUE 行，忽略空行与 # 注释，去首尾引号。"""
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        if key:
            pairs.append((key, value))
    return pairs


def _parse_temperature(raw: str | None) -> float:
    if raw is None or raw.strip() == "":
        return DEFAULT_TEMPERATURE
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"CODING_AGENT_TEMPERATURE 非法: {raw!r}") from exc
