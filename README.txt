# coding-agent

一个轻量级本地编程助手，通过 OpenAI 兼容接口和大模型协作，在指定 workspace 内读取代码、修改文件、运行测试。

适合处理小型编程任务，观察一个 coding agent 如何在本地完成“理解代码、修改文件、运行验证、生成报告”的完整闭环。模型负责判断下一步做什么，本地 runtime 负责工具调用、权限控制、文件系统边界和命令执行。

## 仓库地址

https://github.com/JN-luo/coding-agent

## 功能概览

- 交互式 REPL：可以像命令行助手一样连续提交任务。
- 模型解析：解析模型返回的工具调用，校验参数后交给本地工具执行。
- 本地工具：支持列目录、读文件、glob、grep、写文件和运行白名单命令。
- 权限模式：默认 ask，写文件和运行命令前询问；也支持 readonly 和 auto。
- 上下文管理：会话内保留历史任务、工具结果和最终回答，必要时做语义压缩。
- 调试 trace：每次运行写入 JSONL 日志，方便复盘模型动作和工具结果。

## 安装与配置

依赖：

- Python 3.10+
- 运行测试需要 pytest
- 目标项目的测试命令需要对应环境，例如 Java 项目需要 JDK 和 Maven

配置可以写入环境变量或 `.env`：

```text
OPENAI_API_KEY=模型或网关密钥
OPENAI_BASE_URL=https://api.openai.com/v1
CODING_AGENT_MODEL=模型名
CODING_AGENT_TEMPERATURE=0
```

## 使用方式

单次任务：

```powershell
python -m agent "给 demo/calculator.py 增加 add_many 方法并补测试"
```

进入交互式 REPL：

```powershell
python -m agent
```

指定工作目录：

```powershell
python -m agent --workspace ./demo
```

常用模式：

```powershell
python -m agent --readonly          # 只读浏览，不允许写文件或运行命令
python -m agent --auto              # 自动执行写文件和白名单命令
python -m agent --max-steps 40      # 调整单个任务的最大步数
```

默认模式是 `ask`：读取类工具直接执行，`write_file` 和 `run_command` 会先询问用户。

## 工具

当前内置 6 个本地工具：

- `list_files`：列出目录直接子项。
- `read_file`：读取文本文件，超长内容会截断。
- `glob`：按 glob pattern 查找文件或目录。
- `grep`：按正则搜索文本内容。
- `write_file`：覆盖写入 workspace 内文本文件。
- `run_command`：执行 allowlist 中的测试类命令。

工具层会限制路径必须位于 workspace 内，并拒绝访问 `.env`、私钥等敏感文件。
`run_command` 默认拒绝 shell 元字符、删除命令、安装命令和未知命令。

## 实现说明

项目没有使用外部 agent 框架，核心运行结构如下：

```text
CLI / REPL (cli.py)              解析参数、加载配置，构造 LLM + Session + Tracer
    │
    ▼
Session (session.py)  ──  会话容器，进程内持续运行、跨任务累积上下文
    ├─ Conversation (messages.py)   messages + turns，单会话内跨任务累积
    ├─ Trace (trace.py)             JSONL 调试日志，仅观察、不参与控制流
    └─ submit(task)
         ▼
Agent Loop (loop.py)  ──  单任务闭环：观察 → 思考 → 行动 → 观察
    │
    │  ① Context.build (context.py)   三层上下文，超阈值时语义压缩
    │       │
    │  ② LLM.complete (llm.py)        原生 tool calling 返回工具请求
    │       │
    │  ③ Policy.decide (policy.py)    allow / ask(写操作和执行命令需用户确认) / deny
    │       │
    │  ④ Tool Registry (tools.py)     6 个工具本地执行
    │       │
    │  ⑤ Filesystem / Commands       文件读写 / 命令执行
    │       └── ToolResult 回填 Conversation，回到 ① 继续循环
    │
    ▼   终止条件：模型给出最终回答、达到最大步数、连续同类工具失败、多次触发安全拒绝，或用户多次拒绝高风险操作
Report   步骤数 · 改动文件 · 是否测试 · 遗留问题
```

