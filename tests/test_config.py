"""config.py 的单元测试。

覆盖 DESIGN.md §3 config.py：
  - 从环境变量读取（api_key / base_url / model / temperature）
  - 缺省值（temperature=0.0，其余空串）
  - .env 解析与优先级（环境变量优先）
  - 非法 temperature 抛 ConfigError
"""

from pathlib import Path

import pytest

from agent.config import Config, ConfigError, _merge_dotenv, _parse_dotenv, load_config


def test_load_config_reads_env():
    cfg = load_config(env={
        "OPENAI_API_KEY": "sk-123",
        "OPENAI_BASE_URL": "http://localhost:8000/v1",
        "CODING_AGENT_MODEL": "gpt-4o",
        "CODING_AGENT_TEMPERATURE": "0.3",
    })
    assert cfg == Config(
        api_key="sk-123",
        base_url="http://localhost:8000/v1",
        model="gpt-4o",
        temperature=0.3,
    )


def test_load_config_defaults():
    cfg = load_config(env={"OPENAI_API_KEY": "sk"})
    assert cfg.api_key == "sk"
    assert cfg.base_url == ""
    assert cfg.model == ""
    assert cfg.temperature == 0.0


def test_load_config_empty_env():
    cfg = load_config(env={})
    assert cfg.api_key == ""
    assert cfg.temperature == 0.0


def test_load_config_invalid_temperature():
    with pytest.raises(ConfigError):
        load_config(env={"CODING_AGENT_TEMPERATURE": "abc"})


def test_parse_dotenv_basic(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "# 注释\n"
        "OPENAI_API_KEY=sk-file\n"
        'CODING_AGENT_MODEL="gpt-4o"\n'
        "\n"
        "CODING_AGENT_TEMPERATURE=0\n",
        encoding="utf-8",
    )
    assert _parse_dotenv(p) == [
        ("OPENAI_API_KEY", "sk-file"),
        ("CODING_AGENT_MODEL", "gpt-4o"),
        ("CODING_AGENT_TEMPERATURE", "0"),
    ]


def test_merge_dotenv_env_wins(tmp_path):
    env = {"OPENAI_API_KEY": "from_env"}
    p = tmp_path / ".env"
    p.write_text("OPENAI_API_KEY=from_file\nCODING_AGENT_MODEL=gpt-4o\n", encoding="utf-8")
    _merge_dotenv(env, p)
    assert env["OPENAI_API_KEY"] == "from_env"    # 环境变量优先
    assert env["CODING_AGENT_MODEL"] == "gpt-4o"  # 缺失键被填充


def test_merge_dotenv_missing_file():
    env = {"OPENAI_API_KEY": "x"}
    _merge_dotenv(env, Path("nonexistent.env"))
    assert env == {"OPENAI_API_KEY": "x"}
