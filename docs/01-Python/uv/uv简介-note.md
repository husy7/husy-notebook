---
title: "[uv 简介]"
tags: [python库, 包管理, 内容介绍, uv]
date: 2026-09-03
---
`uv` 是一个用 Rust 编写的、速度极快的 Python 包和项目管理器。它由打造了著名 Python 代码检查工具 Ruff 的 Astral 团队开发，旨在成为一个统一的、高性能的现代 Python 开发工具链。

### 核心亮点：为什么是 uv？

*   **⚡ 极致的速度**：`uv` 的最大卖点就是快。它比传统的 `pip` 和 `pip-tools` 快 **10 到 100 倍**。这得益于其 Rust 语言的高效性能和并行处理能力。官方基准测试显示，在缓存情况下，速度提升甚至可达 **80-115 倍**。
*   **🔧 一站式工具链**：`uv` 旨在用一个工具取代 Python 开发中常用的多个工具，包括 `pip`, `pip-tools`, `pipx`, `poetry`, `pyenv`, `virtualenv` 等。它集成了以下功能：
    *   **依赖管理**：类似 `pip`/`poetry`
    *   **虚拟环境**：类似 `venv`/`virtualenv`
    *   **Python 版本管理**：类似 `pyenv`
    *   **项目管理**：类似 `poetry`
    *   **运行脚本和工具**：类似 `pipx`
*   **🔒 可靠的依赖锁定**：`uv` 使用 `uv.lock` 文件来精确锁定项目依赖的版本，确保在不同环境下安装的依赖完全一致，解决了 `pip` 的 `requirements.txt` 不够精确的问题。
*   **🌍 广泛的兼容性**：`uv` 提供了与 `pip` 兼容的接口 (`uv pip`)，方便用户从旧工作流迁移。同时，它也支持 macOS、Linux 和 Windows 全平台。

### 核心功能与用法速览

#### 1. 安装与 Python 版本管理
安装 `uv` 非常简单，甚至不需要预先安装 Python。
```bash
# macOS/Linux 使用官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 Homebrew (macOS)
brew install uv

# Windows 使用 PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
`uv` 可以直接安装和管理不同版本的 Python。
```bash
# 列出可用的 Python 版本
uv python list

# 安装 Python 3.12
uv python install 3.12
```

#### 2. 项目初始化与环境管理
`uv` 简化了新项目的启动流程。
```bash
# 创建一个新项目，会生成 pyproject.toml 和 .python-version 文件
uv init my-project

# 进入项目目录
cd my-project

# 自动创建 .venv 虚拟环境并安装所有依赖
uv sync
```

#### 3. 依赖管理
你可以像使用 `poetry` 一样方便地管理依赖。
```bash
# 添加项目依赖，自动更新 pyproject.toml 和 uv.lock
uv add requests

# 添加开发依赖
uv add --dev pytest

# 移除依赖
uv remove requests
```

#### 4. 运行脚本和工具
无需手动激活虚拟环境，`uv run` 会自动在项目环境中执行命令。
```bash
# 在项目虚拟环境中运行 Python 脚本
uv run python script.py

# 运行一个一次性工具（如 ruff），用完即删，类似 pipx
uvx ruff check .
```

### 与旧工具的对比

| 功能领域 | 传统工具 | `uv` 的替代方案 |
| :--- | :--- | :--- |
| **Python 版本管理** | `pyenv` | `uv python install` |
| **创建虚拟环境** | `venv`, `virtualenv` | `uv venv` |
| **安装包** | `pip` | `uv pip install` 或 `uv add` |
| **依赖锁定** | `pip-tools`, `poetry` | `uv.lock` |
| **项目管理** | `poetry` | `uv init`, `uv sync` |
| **运行命令行工具** | `pipx` | `uvx`, `uv tool install` |

更详细的信息和高级用法，可以查阅其官方文档：
*   [uv 官方文档 (英文)](https://docs.astral.sh/uv/)
*   [uv 官方文档 (中文)](https://docs.astral.org.cn/uv/)