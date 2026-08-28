# Coding Agent 运行范式设计

## 1. 目标定位

本项目实现一个最小但完整的编程智能体：用户输入编程任务后，agent 可以在本地仓库中读取文件、修改文件、执行命令、观察结果，并通过多轮 LLM 推理逐步完成任务。

设计重点不是堆功能，而是让评委能清楚看到：核心 agent loop、工具协议、上下文管理、错误处理和终止条件都由项目自行实现，且满足题目的硬性合规约束——不用任何 agent 框架/SDK、不依赖 API 服务端托管的代码执行或文件工具、凭据不入库。

## 2. 总体运行范式

采用 `Observe -> Think -> Act -> Observe` 的闭环：

1. 用户给出任务。
2. Agent 将任务、系统约束、仓库摘要、历史步骤压缩为 prompt。
3. LLM 输出结构化动作，动作只允许是预定义工具调用或最终回答。
4. Runtime 校验动作合法性，然后在本地执行工具。
5. 工具结果回填到对话历史。
6. Agent 判断是否继续、重试、压缩上下文或终止。

**协议约定：核心版本每轮只输出一个动作**（无论只读还是写/执行），与 §3 `parser.py` 的单 action JSON schema 保持一致。只读批量作为可选增强见 §8.5，不在核心版本实现，避免出现"说了但没定义"的歧义。

## 3. 模块划分

建议仓库结构：

```text
coding-agent/
  agent/
    __init__.py
    cli.py
    loop.py
    llm.py
    messages.py
    tools.py
    context.py
    parser.py
    config.py
  tests/
    test_parser.py
    test_tools.py
    test_context.py
  README.txt
  DESIGN.md
  .gitignore
```

### `cli.py`

负责命令行入口：

```bash
python -m agent "修复测试失败"
python -m agent --workspace ./demo "给项目加一个 argparse 参数"
```

它只处理参数、加载配置、启动 loop，不放复杂逻辑。

### `loop.py`

核心调度器，维护 agent 状态：

- 当前任务
- 消息历史
- 已执行步骤数
- 最近工具结果
- 错误计数
- token 或字符预算
- 是否完成

伪代码：

```python
while step < max_steps:
    prompt = context.build(task, history, workspace_summary)
    raw = llm.complete(prompt, tool_schema)
    action = parser.parse(raw)

    if action.type == "final":
        return action.message

    result = tools.run(action)
    history.append(action, result)

    if should_stop(history, result):
        return summarize_result(history)
```

### `llm.py`

只封装模型调用。允许使用模型厂商 API 客户端，但不依赖托管代码执行、文件工具或 agent SDK。

配置来源：

- `OPENAI_API_KEY` 或兼容网关 key
- `OPENAI_BASE_URL`
- `CODING_AGENT_MODEL`
- `CODING_AGENT_TEMPERATURE`（演示时建议设为 0，见 §9）

`config.py` 只从环境变量或 `.gitignore` 排除的 `.env` / `config.local.*` 读取；仓库根目录提供 `.gitignore`，排除任何可能含凭据的文件。README 中明确"若曾误提交凭据，立即作废更换"。

### `parser.py`

负责把 LLM 输出解析为动作。采用 JSON 输出而非模型原生 tool calling——这最大化"模型输出解析"的自研成色（题目把解析列为必须自研项），也让解析逻辑可测试、可讲解：

```json
{
  "thought": "需要先查看项目结构",
  "action": "list_files",
  "args": {
    "path": "."
  }
}
```

最终回答格式：

```json
{
  "action": "final",
  "message": "已完成修改，并通过测试。"
}
```

解析器必须处理以下五种边界，且**每一条都必须写进 `test_parser.py`**（这是最重要的自研单元）：

- JSON 外包裹 Markdown 代码块
- JSON 缺字段
- 未知工具名
- 参数类型错误
- 模型输出自然语言时的兜底报错

### `tools.py`

工具层参考 OpenCode 精简为「一个接口 + 一个结果类型」，安全约束内聚为 helper，不再单列 `sandbox.py`：

```python
@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str      # 成功=给模型看的结果；失败=错误信息
    error: str = ""  # 结构化错误类型：FileNotFound / PolicyDenied / Timeout ...

@dataclass(frozen=True)
class Tool:
    name: str
    description: str   # 给模型看的自然语言说明
    parameters: dict   # JSON Schema，喂给模型
    execute: Callable[..., ToolResult]

TOOLS: dict[str, Tool] = {}   # 注册表；喂给模型的工具说明由这里生成
```

工具统一返回 `ToolResult`，loop 只做 `history.append(result)`，不感知具体工具。失败分两类：预期失败抛 `ToolError(type, message)`，由 `run_tool` 统一转成 `ToolResult(ok=False)`；未预期异常兜底为 `ToolResult(ok=False)`（error 取异常类名），绝不让 loop 崩溃。

第一版 6 个工具（OpenCode 风格，拆出 `glob` / `grep`）：

| 工具 | 作用 | 风险 |
| --- | --- | --- |
| `list_files` | 列出目录内容 | 低 |
| `read_file` | 读取文本文件（100 KB 上限） | 低 |
| `glob` | 按文件名 pattern 匹配（如 `**/*.py`） | 低 |
| `grep` | 按正则搜索文件内容 | 低 |
| `write_file` | 覆盖写入文件 | 高 |
| `run_command` | 执行命令（30s 超时、截断 12k 字符） | 高 |

安全约束（作为本文件的 helper，替代原 `sandbox.py`）：

- `resolve_in_workspace(path, workspace)`：resolve 后必须落在 workspace 内，越权抛 `PathOutsideWorkspace`。
- `check_command_policy(command)`：先拒 shell 元字符（`;` / `&&` / `|` / 反引号 / `$(`），再命中前缀白名单（`python` / `npm test` / `cargo test` / `go test`），未命中白名单则落回黑名单（`rm` / `del` / `pip install` …）拒绝，最终默认拒绝。MVP 只做 allow / deny。
- 凭据保护：`read_file` / `write_file` 拒绝访问 `.env`、私钥（`id_rsa` / `*.pem` / `*.key`）；`grep` 静默跳过这些文件。
- 跨平台：命令执行用 `subprocess.run(shell=False)`（配 `shlex.split` 拆词，安全边界由白名单策略承担）；路径用 `pathlib`。开发环境为 Windows，6 个工具须在 Windows 本机跑通才算 MVP 完成。

### `context.py`

上下文管理是面试重点。三层上下文 + 压缩，**压缩是核心自研项，不是可选增强**：

1. **固定系统规则**：agent 能做什么、不能做什么、工具 JSON 格式。
2. **工作区摘要**：项目语言、主要文件、最近读过的文件摘要。
3. **短期轨迹**：最近 N 轮 action/result 原文。

当历史超过阈值（字符数或轮数）时，在 core 里触发压缩，不直接丢弃全部内容，而是把旧轨迹压缩为：

```text
已完成：
- 读取了 pyproject.toml，确认项目使用 pytest
- 修改了 agent/parser.py，增加 JSON 代码块解析

仍需：
- 运行测试
- 如果失败，修复 parser 边界情况
```

压缩必须保证"不丢失未完成事项与关键结论"，并配 `test_context.py` 验证压缩前后信息保留。

### `messages.py`

维护发给模型的消息列表（system + user + assistant + tool 结果），负责把 action/result 转成下一轮的 assistant/tool 消息，是"对话历史管理"的落点。

## 4. 推荐的 Prompt 协议

系统提示词核心内容：

```text
你是一个本地 coding agent。你只能通过工具观察和修改仓库。
每次回复必须是一个 JSON 对象。
如果任务完成，使用 action=final。
不要编造文件内容；需要信息时先 read_file 或 grep。
写文件前尽量先读取目标文件。
执行命令失败时，先根据 stderr 分析，再决定下一步。
```

工具 schema 由代码生成，而不是手写散落在 prompt 中，避免实现和说明不一致。

## 5. 循环终止条件

至少实现以下终止条件：

- `final` 动作：模型主动完成。
- `max_steps`：默认 20 或 30 步。
- 连续解析失败：例如 3 次。
- 连续同类工具失败：例如同一命令失败 3 次。
- 安全拒绝：模型多次请求越权路径或危险命令。

终止时输出简洁报告：

- 做了什么
- 修改了哪些文件
- 是否运行测试
- 测试结果
- 未完成事项

## 6. 错误处理策略

错误不能只抛异常结束，要反馈给模型形成闭环：

```json
{
  "tool": "read_file",
  "ok": false,
  "error_type": "FileNotFound",
  "message": "path not found: src/main.py"
}
```

模型下一轮应该根据错误换路径、搜索文件或请求用户澄清。

## 7. 最小可行版本

第一阶段只做命令行单 agent：

1. CLI 接收任务。
2. LLM 返回 JSON 动作。
3. 支持 `list_files`、`read_file`、`glob`、`grep`、`write_file`、`run_command`（6 个）。
4. 路径限制在 workspace 内（含 Windows 兼容）。
5. 最多运行 20 步。
6. 历史超阈值触发压缩（§3 三层上下文 + 压缩，核心项）。
7. 危险命令直接拒绝并回填错误。
8. 最终输出任务报告。

这个版本已满足项目要求中的核心逻辑自研：对话历史、上下文、工具定义、本地执行、输出解析、终止条件、错误处理。

## 8. 可作为特色功能的增强

按实现成本与面试可解释性排序：

1. **变更报告**：结束时列出 modified files 和测试命令（成本最低，先做）。
2. **Dry run 模式**：只展示计划和将要调用的工具，不真正写文件。
3. **命令 ask 分级**：在 allow / deny 之外加 ask，对未知命令要求用户确认。
4. **apply_patch 差异修改（可选，慎做）**：模型输出 unified diff，runtime 校验后应用。unified diff 的模糊匹配、行号漂移是已知工程坑；若做，建议用成熟 diff 库（如 `unidiff`，非 agent SDK，合规），并放到最后，避免卡进度。
5. **只读批量**：一轮输出 `{"action": "batch", "actions": [{...}, {...}]}`，只允许低风险只读工具，用于列目录 + 读多文件提速。
6. **更智能的上下文压缩**：在 §7 的阈值压缩之上，做语义级"已完成/仍需"摘要（core 已含基础版，这里是增强）。

## 9. 演示任务建议

视频中选择一个小而完整的真实任务：

```text
给这个 Python 项目增加一个 Calculator.add_many(numbers) 方法，
补充单元测试，并运行测试确认通过。
```

演示路径：

1. Agent 列目录。
2. 读取源码和测试。
3. 修改源码。
4. 修改测试。
5. 运行测试。
6. 根据测试结果修正或最终报告。

这个任务足够真实，又不会在 2 分钟视频里失控。

**演示可靠性优先于功能丰富度**（视频 ≤2 分钟、要一次成功）：

- 固定用强模型 + 低 temperature（`CODING_AGENT_TEMPERATURE=0`），减少输出漂移。
- 同一任务至少预演 5 次，确认演示路径稳定。
- 准备"模型跑偏时人工兜底"的录制方案（重录或剪辑）。

## 10. 提交物与时间线

### 10.1 三件提交物

1. **Git 仓库**：题目发布后新建的公开仓库（GitHub/Gitee）；保留完整提交历史，不得压缩或改写已推送历史；**9 月 2 日 24:00 后不再 push**；地址写在 README.txt 中。
2. **README.txt（≤1000 汉字）**：必须含——仓库地址；如何运行（依赖、环境变量、启动命令）；特色功能说明（重点写自研的 parser / context / sandbox / 终止条件）。
3. **视频（≤2 分钟，mp4，≤200 MB）**：演示 agent 完成真实编程任务 + 简要讲解实现；允许剪辑与加速。

提交物打包为 `你的姓名.zip`，内含 README.txt 与视频。

### 10.2 关键时间线

- 仓库新建：题目发布后（保留创建时间作为"新仓库"证明）。
- MVP 完成 + demo 预演：8 月底前。
- 视频录制 + 剪辑：9 月 1 日前。
- 最终检查（README 字数、仓库历史完整、key 未泄露）：9 月 2 日前。
- 9 月 2 日 24:00 后：只读，不再推送。

### 10.3 凭据安全（规则 4 落地）

- 所有 key 只走环境变量，或 `.gitignore` 排除的 `.env` / `config.local.*`。
- 提交前用 `git log -p` 或 `git grep` 检查历史中无 key 泄露；若曾误提交，立即作废更换（作废比删除更重要）。
