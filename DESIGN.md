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

**协议约定：采用模型原生 tool calling**（见 §3 `llm.py` / `policy.py`）。每轮 LLM 响应可返回**多个 `tool_calls`**（浏览阶段批量只读，写/命令逐个裁决）；无 `tool_calls` 且带 content 时视为 final。参数校验、权限裁决、本地执行、错误回填都在本地 runtime 完成，不依赖模型端托管执行。

## 3. 模块划分

### 会话架构（Session / Conversation / Trace）

核心版本只做**持续交互**，不做恢复 / 回放。目标是「一个进程里像 Claude Code 一样一直聊下去」：

- **Session**（`session.py`）：REPL 运行时容器。持有 workspace、llm、Conversation、Trace、历史 Report。一个 `python -m agent` 进程对应一个 Session，直到退出为止。
- **Conversation**（`messages.py`）：真正发给模型的持续上下文。它是活的、内存中的对话序列，跨任务累积，但不落盘、不负责恢复。
- **Trace**（`trace.py`）：仅作调试日志。append-only，方便看每一步发生了什么，但**不参与控制流，不作为恢复源**。

```text
CLI / REPL
  -> Session（进程内长期存在）
      -> Conversation（持续上下文）
      -> loop step（单任务 Observe→Think→Act→Observe）
      -> Trace（可选日志）
```

原则：

1. **单任务状态与跨任务状态分离**：safety / same_tool / user_denies / max_steps 每任务重置；conversation / workspace 跨任务连续；trace 只是日志。
2. **Conversation 是唯一的活上下文**：它只负责给模型看的消息序列，不负责调度。
3. **context 从 Conversation 取数**：新任务时从 session.conversation 构建 prompt，保留系统提示词、历史任务、工具结果与压缩摘要。
4. **Report 是每任务输出**：Session 聚合多个 Report，但不把 Report 当作状态源。

建议仓库结构：

```text
coding-agent/
  agent/
    __init__.py
    __main__.py
    cli.py
    session.py
    loop.py
    llm.py
    messages.py
    trace.py
    tools.py
    prompts.py
    policy.py
    context.py
    parser.py
    config.py
  tests/
    test_parser.py
    test_tools.py
    test_messages.py
    test_trace.py
    test_context.py
    test_prompts.py
    test_config.py
    test_llm.py
    test_loop.py
    test_session.py
  README.txt
  DESIGN.md
  .gitignore
```

### `cli.py`

REPL 外壳，维护一个长期 session：

```bash
python -m agent "任务"                    # one-shot：单任务 session，跑完即退
python -m agent                           # 无任务 → 进 REPL，长期 session
python -m agent --workspace ./demo ...    # workspace 缺省为当前目录 Path.cwd()
python -m agent --max-steps 40 ...        # 覆盖单任务最大步数，便于演示调参
```

它只做参数解析、`config.load_config`、构造 `Session`、REPL 循环（`input` 读任务、`session.submit`、打印 Report），不放复杂逻辑。`--max-steps` 只影响单个 task，跨任务时每轮重新计数。

### `loop.py`

单任务执行器（`Observe -> Think -> Act -> Observe`），由 Session 驱动。它只负责完成一个 task 的推理闭环，不保存跨任务状态。

单任务状态（每任务重置）：step、连续同类工具失败、安全拒绝、用户拒绝计数、task 级授权。伪代码：

```python
def run_task(session, task) -> Report:
    session.trace.log("task", text=task)
    task_grants = set()
    while step < max_steps:
        messages = context.build(session.conversation, task)
        response = session.llm.complete(messages, tools=tools_schema)
        session.trace.log("llm_response", n_tool_calls=len(response.tool_calls))

        if not response.tool_calls:            # 无 tool_calls → final
            return report(done=True, message=response.content, ...)

        session.conversation.append_assistant_tool_calls(response)
        for tc in response.tool_calls:         # 一轮可多个工具
            decision = policy.decide(session.mode, tc.name)
            if tc.parse_error:                 # 参数 JSON 非法 → InvalidArgs
                result = ToolResult(ok=False, error="InvalidArgs", output=tc.parse_error)
            elif decision.deny:
                result = ToolResult(ok=False, error="PolicyDenied", output=decision.reason)
            elif decision.ask:
                key = policy.grant_key(tc.name, tc.arguments)
                if key not in task_grants:
                    choice = session.asker(tc.name, tc.arguments)  # once / remember / deny
                    if choice == "deny":
                        result = ToolResult(ok=False, error="UserDenied", output="用户拒绝执行该动作")
                    elif choice == "remember":
                        task_grants.add(key)
                        result = run_tool(tc.name, tc.arguments, session.workspace)
                    else:  # once
                        result = run_tool(tc.name, tc.arguments, session.workspace)
                else:
                    result = run_tool(tc.name, tc.arguments, session.workspace)
            else:
                result = run_tool(tc.name, tc.arguments, session.workspace)

            session.conversation.append_tool_result(tc.id, tc.name, tc.arguments, result)
            session.trace.log("tool", tool=tc.name, ok=result.ok, error=result.error)

            if should_stop(...):
                return report(...)
```

### `session.py`

跨任务状态与编排的核心对象：

- 持有 `Conversation`、`Trace`、`workspace`、`llm`、`reports: list[Report]`。
- 持有 `mode`（ask / readonly / auto）与 `asker` 回调（CLI 注入的 once / remember / deny 交互询问）。
- `submit(task) -> Report`：把新 task 交给 loop 跑完，然后把结果写回 conversation + trace，保存 Report。
- 支撑 REPL：一个 session 跑到底，直到用户退出才销毁 conversation。

这是「像 Claude Code 连续感」的落点：单任务计数归 loop，跨任务连续性归 Session。

### `llm.py`

只封装模型调用，提供 `complete(messages, tools) -> ModelResponse`（主对话，原生 tool calling，返回 content 或 tool_calls）与 `summarize(turns) -> str`（语义压缩，独立非 tool-calling 路径）。允许使用模型厂商 API 客户端，但不依赖托管代码执行、文件工具或 agent SDK；HTTP 用标准库 `urllib`。

配置来源：

- `OPENAI_API_KEY` 或兼容网关 key
- `OPENAI_BASE_URL`
- `CODING_AGENT_MODEL`
- `CODING_AGENT_TEMPERATURE`（演示时建议设为 0，见 §9）

`config.py` 只从环境变量或 `.gitignore` 排除的 `.env` / `config.local.*` 读取；仓库根目录提供 `.gitignore`，排除任何可能含凭据的文件。README 中明确"若曾误提交凭据，立即作废更换"。

### `parser.py`

已收窄为 legacy fallback：主路径改用模型原生 tool calling（LLM 直接返回结构化 `tool_calls`，参数 JSON 由 `llm` 解析），不再从文本抠 JSON。本模块保留旧 JSON 文本 action 解析（含 `test_parser.py` 的五种边界测试），仅在需要兼容不支持 tool calling 的模型时作为 fallback。

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
- `check_command_policy(command)`：先拒引号外的 shell 操作符（`;` / `&&` / `|` / 反引号 / `$(`，引号内是字符串内容不算），再命中前缀白名单（`python` / `pytest` / `mvn test` / `mvn -q test` / `npm test` / `cargo test` / `go test`），未命中白名单则落回黑名单（`rm` / `del` / `pip install` …）拒绝，最终默认拒绝。
- 凭据保护：`read_file` / `write_file` 拒绝访问 `.env`、私钥（`id_rsa` / `*.pem` / `*.key`）；`grep` 静默跳过这些文件。
- 噪声目录过滤：`list_files` / `glob` / `grep` 默认跳过 `.git`、`.venv`、`__pycache__`、`.pytest_cache`、`node_modules`、`target`、`dist`、`build` 等缓存、依赖和构建产物；显式 `read_file` / `write_file` 这些目录内文件时返回 `IgnoredPath`，减少上下文污染和无意义探索。
- 跨平台：命令执行用 `subprocess.run(shell=False)`（配 `shlex.split` 拆词，安全边界由白名单策略承担）；路径用 `pathlib`。开发环境为 Windows，6 个工具须在 Windows 本机跑通才算 MVP 完成。

### `policy.py`

权限层位于 parser 和 tools 之间：LLM 只负责提出动作，runtime 根据任务模式裁决 `allow / ask / deny`。这层是 prompt 约束的硬化版本，避免"模型说要写就直接写"。

三种运行模式：

| 模式 | 读工具 | `write_file` | `run_command` |
| --- | --- | --- | --- |
| `ask`（默认） | allow | ask | ask |
| `readonly` | allow | deny | deny |
| `auto` | allow | allow | allow（仍受命令白名单限制） |

模式来源：

- CLI 默认 `ask`。
- `--readonly` 强制整个 session 只读：任何授权都不能升级它。
- `--auto` 用于演示或用户明确信任：写文件与白名单命令自动执行。

ask 层只在当前任务内记录用户选择，不做持久化、不做恢复：

```text
y / yes     仅允许本次动作
a / always  允许本次动作，并在本任务内记住该授权
n / no      拒绝本次动作（默认）
```

- 会话级放行不在 ask 里，改用 `--auto` 启动时声明，避免中途反复确认。
- 记住授权是**任务级**的（每任务重置）：换一个任务，写/跑要重新确认。
- 拒绝不会写入长期 deny 缓存，而是作为 `UserDenied` observation 回填给模型；模型可以换路径、换命令或给出无法继续的说明。

授权粒度：

- `write_file`：`remember` key 为 `write_file`。用户选择记住后，本任务内后续写文件免确认；用户选择 `once` 时只允许当前这次写入。
- `run_command`：`remember` key 为 `run_command:<command>`，按完整命令字符串授权。即使用户选择记住，也只允许本任务内重复执行同一条命令；`pytest -q`、`pytest -q tests/test_cli.py`、`mvn test`、`mvn -q test` 都是不同授权。
- `readonly` 模式不读取任何授权，始终禁止 `write_file` 与 `run_command`。

裁决顺序：

1. LLM 返回结构化 `tool_calls`，参数 JSON 由 `llm` 解析（非法 JSON → `parse_error` → InvalidArgs 回填）；loop 校验工具名是否存在。
2. `policy.decide(mode, action)` 判断当前模式下是 allow、ask 还是 deny。
3. deny：不执行工具，合成 `ToolResult(ok=False, error="PolicyDenied", output=reason)` 回填给模型。
4. ask：若当前 task 已记住该授权则直接执行；否则 CLI 询问用户 `once / remember / deny`。`once` 只执行本次，`remember` 执行并记住到当前 task 结束，`deny` 不执行并回填 `UserDenied`。
5. allow：再进入 `tools.run_tool`，由工具层执行路径、敏感文件、命令白名单等底线校验。

这使安全边界分层清楚：prompt 负责引导，policy 负责模式权限，tools 负责底层沙箱。

### `prompts.py`

系统提示词单独放在这里，负责输出协议、工具白名单、workspace / 安全边界和失败恢复规则。
`context.py` 只负责从 conversation 装配消息（系统提示词 + 各任务 + 轨迹），超阈值时触发语义压缩。

### `context.py`

上下文管理是面试重点。三层上下文 + 语义压缩，**压缩是核心自研项，不是可选增强**：

1. **固定系统规则**：agent 能做什么、不能做什么、工具 schema（由 `TOOLS` 派生，喂给模型的原生 tool calling 定义）。
2. **工作区摘要**：项目语言、主要文件、最近读过的文件摘要——由 conversation 里的历史观察（list_files/read_file）自然体现，不单独计算。
3. **短期轨迹**：最近 N 轮 action/result 原文。

输入从「裸 turns」改为 **conversation**：新任务时从 `session.conversation.turns` 提取活跃轨迹（system + 各任务 + 轨迹），超阈值时把旧轨迹交给语义 summarizer 压缩，保留未完成事项和关键结论。`context.py` 只负责从 conversation 装配和触发压缩，不做规则摘要。

语义 summarizer 由 `llm.summarize` 提供；`test_context.py` 用 fake summarizer 验证接口形状与信息保留。

### `messages.py`

conversation 核心，三件套：`Message(role, content, tool_calls, tool_call_id, name)` + `Turn(action, args, result)` + `Conversation`。`Conversation` 同时持有 `messages`（发给模型的序列，支持原生 tool calling 结构）与 `turns`（结构化轨迹，供 context 压缩），是**活上下文**，跨任务累积，不落盘、不恢复。

消息角色：

- `system`：固定规则 + 工具 schema（由 `prompts.py` 构建，进程开头写一次）。
- `user`：每轮任务（每个 task 一条）。
- `assistant`：模型当轮的 `tool_calls`（或 final 的纯文本 content）。
- `tool`：单个工具的执行结果（带 `tool_call_id` 与 `name`）。

接口：`set_system(rules)`、`add_task(text)`、`append_assistant_tool_calls(response)`（追加 assistant 的 tool_calls 消息，保留 raw arguments 与 reasoning_content）、`append_tool_result(tool_call_id, name, args, result)`（追加 tool 结果消息 + 一条 Turn）、`add_final(content)`（追加 final 结论）、`as_openai()`、`total_chars()`。工具结果正文对齐 §6——成功 `{"tool", "ok", "output"}`、失败 `{"tool", "ok", "error_type", "message"}`。

### `trace.py`

落盘的调试日志，只做观察，不参与控制流：

- 每行一个事件，`flush()` 每行，崩溃时尽量保住已写行。
- 关键事件：`task`、`llm_response`（n_tool_calls）、`tool`（ok / error / output_len / output_preview）、`summarize_error`、`stop`。失败工具的 output 记录截断后的正文（前 1000 字符），避免只看 output_len 无法定位失败原因。
- 文件按 session 命名 `run-<时间戳>.jsonl`，目录由 `CODING_AGENT_TRACE_DIR` 配置（默认 `trace/`），加入 `.gitignore`。
- 只做 append-only 日志，不做 replay / resume / 恢复。

## 4. 推荐的 Prompt 协议

系统提示词（`prompts.py` 生成规则；工具定义不再手写进 prompt，而是通过请求的 `tools` + `tool_choice` 参数传给模型）：

```text
你是一个本地 coding agent，在 workspace 目录内观察、修改代码，并运行测试验证。
使用提供的工具（tool calling）观察和修改仓库；信息不足时先 read_file / list_files / glob / grep。

【路径】所有 path 都相对 workspace 根目录，不能访问 workspace 之外。

【原则】
  - 不要编造文件内容；需要信息时先搜索。
  - 写文件前尽量先读取目标文件。
  - 执行命令失败时先分析 stderr。
  - 任务完成后停止调用工具，直接给出最终回答。
```

工具 schema 由代码生成（`Tool.to_schema()` 从 `TOOLS` 派生，随请求传入），避免实现和说明不一致。

## 5. 循环终止条件

以下终止条件都是**单任务内**的（每任务重置；Session 跨任务存在）：

- `final`：模型不再调用工具、返回纯文本回答。
- `max_steps`：默认 50 步，可由 CLI 的 `--max-steps` 覆盖。
- 连续同类工具失败：例如同一命令失败 3 次。
- 安全拒绝：模型多次请求越权路径或危险命令（PolicyDenied / PathOutsideWorkspace）。
- 用户拒绝：用户拒绝会先回填 `UserDenied` 让模型自我修正；若连续拒绝 3 次、累计拒绝 5 次，或同类高风险动作（`write_file` / `run_command`）被拒绝 3 次，则提前停止。

终止时输出简洁报告：

- 终止原因（`stop_reason`：final / max_steps / same_tool_failures / safety_rejections / user_denied / llm_error）
- 做了什么
- 修改了哪些文件
- 是否运行测试
- 测试结果
- 过程中遇到的问题（pending）

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

第一阶段做命令行持续交互 agent（`python -m agent` 进入 REPL，`python -m agent "任务"` 仍保留 one-shot）：

1. CLI/REPL 接收任务。
2. `Session` 维持一个长期 `Conversation`。
3. LLM 通过原生 tool calling 返回结构化 `tool_calls`（一轮可多个）。
4. 支持 `list_files`、`read_file`、`glob`、`grep`、`write_file`、`run_command`（6 个）。
5. 路径限制在 workspace 内（含 Windows 兼容）。
6. 每个 task 最多运行 50 步（可 `--max-steps` 覆盖）。
7. 历史超阈值触发语义压缩（§3 三层上下文 + summarizer）。
8. 模式 + ask 层：`ask / readonly / auto`，ask 下写/命令询问用户。
9. 危险命令直接拒绝并回填错误。
10. 每个 task 输出一份 Report，REPL 继续等下一条任务。

这个版本已满足项目要求中的核心逻辑自研：对话历史、上下文、工具定义、本地执行、输出解析、终止条件、错误处理。

## 8. 可作为特色功能的增强

按实现成本与面试可解释性排序：

1. **Dry run 模式**：只展示计划和将要调用的工具，不真正写文件。
2. **apply_patch 差异修改（可选，慎做）**：模型输出 unified diff，runtime 校验后应用。unified diff 的模糊匹配、行号漂移是已知工程坑；若做，建议用成熟 diff 库（如 `unidiff`，非 agent SDK，合规），并放到最后，避免卡进度。

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
