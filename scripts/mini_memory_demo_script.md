# Mini Memory Filesystem 演示脚本（小版本）

目标：用干净的 Java Maven 项目演示 agent 能先理解仓库，再分两轮完成一个增量功能：先改源码和命令入口，再补测试并回归验证。任务选 `IFIND`，比 `MV` 小很多，但仍然不是 toy example。

## 准备

先确保 Maven 使用 JDK：

```powershell
$env:JAVA_HOME='C:\Program Files\Java\jdk-17.0.12'
$env:Path="$env:JAVA_HOME\bin;$env:Path"
```

启动：

```powershell
cd "D:\coding agent\coding-agent"
.\.venv\Scripts\python.exe -m agent --workspace mini-memory-filesystem --auto --max-steps 40
```

如果还没有实现 `--max-steps`，先用默认命令也可以：

```powershell
.\.venv\Scripts\python.exe -m agent --workspace mini-memory-filesystem --auto
```

## 第一轮：快速浏览仓库

输入：

```text
请快速浏览这个项目，概括项目类型、核心模块和测试方式；只做必要阅读，不要修改文件或运行测试。
```

预期：

```text
list_files
read_file pom.xml
read_file architecture.md
read_file Main.java
read_file FileSystem.java
final
```

## 第二轮：新增 IFIND 命令

输入：

```text
基于刚才浏览的结构，为这个内存文件系统新增 IFIND 命令：用法和 FIND 一样，但按名称查找时忽略大小写。只完成源码和命令入口的必要修改。
```

为什么这个任务合适：

- 真实项目：Java 17 + Maven + JUnit 5 + 标准源码/测试目录。
- 改动跨两处：`FileSystem.java`、`Main.java`。
- 功能有业务语义：在现有 `FIND` 上扩展大小写不敏感查找。
- 难度适中：可复用现有递归搜索，不涉及复杂移动/覆盖/链接所有权语义。

预期步骤：

```text
read_file src/main/java/main/FileSystem.java
read_file src/main/java/main/Main.java
write_file src/main/java/main/FileSystem.java
write_file src/main/java/main/Main.java
run_command mvn -q test
final
```

## 第三轮：补测试并验证

输入：

```text
刚才已经新增了 IFIND。请现在为它补充必要的 JUnit 测试，最后运行 mvn -q test 确认全部通过。
```

这一轮展示 Session 能承接上一轮改动，同时把“补测试”和“回归验证”单独拿出来讲。这样演示节奏更自然：先交付最小功能，再补质量保障。

预期步骤：

```text
read_file src/test/java/main/FileSystemFINDTest.java
read_file src/test/java/main/FileSystemIntegrationTest.java
write_file src/test/java/main/FileSystemFINDTest.java
write_file src/test/java/main/FileSystemIntegrationTest.java
run_command mvn -q test
final
```

## 讲解词

开场：

```text
这里我用一个 Java Maven 内存文件系统仓库做演示。Agent 每轮只输出一个 JSON action，本地 runtime 负责解析、权限裁决、执行工具，并把观察结果回填给模型。
```

收尾：

```text
这次改动没有使用 agent 框架，也没有托管代码执行。Agent 自己读取 Maven 项目结构，先修改核心业务类和命令入口，再在下一轮补 JUnit 测试，并在本地运行 mvn -q test 验证。
```

## 如果模型跑偏

如果它开始读很多无关文件，直接中断重跑，并把第二轮任务改得更硬一点：

```text
第二轮只读取 src/main/java/main/FileSystem.java 和 src/main/java/main/Main.java。不要新增测试。实现 IFIND 后立即运行 mvn -q test。
```
