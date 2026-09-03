---
title: "[uv python虚拟环境管理工具]"
tags: [python库, 包管理, 虚拟环境, uv]
date: 2026-09-03
---

### 核心特性：默认且自动化的虚拟环境

`uv` 的一个核心设计理念是**默认要求并使用虚拟环境**。这与 `pip` 默认安装到系统环境的做法不同，有助于从一开始就保持项目依赖的隔离。

*   **无需手动激活**：在传统的 `venv` 工作流中，你需要先激活虚拟环境 (`source .venv/bin/activate`)，然后才能安装包或运行脚本。而 `uv` 会自动发现并使用项目根目录下的 `.venv` 文件夹。这意味着你**不需要手动激活**，直接运行 `uv run`、`uv sync` 等命令即可。

### 核心操作与实践

#### 1. 创建虚拟环境

使用 `uv venv` 命令创建虚拟环境。默认情况下，它会在当前工作目录创建一个名为 `.venv` 的文件夹。

```bash
# 创建默认的 .venv 虚拟环境
uv venv
```

你也可以指定自定义的路径或名称：
```bash
# 创建一个名为 'my-name' 的虚拟环境
uv venv my-name
```

#### 2. 指定 Python 版本

`uv` 可以轻松地为虚拟环境指定 Python 版本。如果系统中没有该版本，`uv` 会自动下载并安装。

```bash
# 使用 Python 3.11 创建虚拟环境
uv venv --python 3.11

# 使用 Python 3.12.0 创建虚拟环境
uv venv --python 3.12.0
```

#### 3. 在项目中使用虚拟环境

当你在一个有 `pyproject.toml` 的项目中工作时，`uv` 的管理更加自动化。

*   **自动创建**：当你首次运行项目命令（如 `uv run`、`uv sync`）时，`uv` 会自动在项目根目录下创建 `.venv` 虚拟环境和 `uv.lock` 锁定文件。
*   **依赖同步**：`uv sync` 命令会根据 `pyproject.toml` 和 `uv.lock` 文件，将项目所有依赖安装到 `.venv` 虚拟环境中。
*   **运行脚本**：使用 `uv run` 命令可以直接在项目的虚拟环境中执行脚本或命令，无需手动激活环境。

### 与现有工具的对比

`uv` 旨在成为一个统一的、更快的替代品，涵盖了多个工具的功能。

| 功能领域 | 传统工具链 | `uv` 的替代方案 |
| :--- | :--- | :--- |
| **创建虚拟环境** | `venv`, `virtualenv` | `uv venv` |
| **管理包** | `pip`, `pip-tools` | `uv pip`, `uv add`, `uv remove` |
| **管理 Python 版本** | `pyenv` | `uv python install` |
| **项目管理** | `poetry` | `uv init`, `uv sync`, `uv run` |

### 高级用法

*   **临时环境**：`uv run --isolated` 命令会创建一个临时环境来运行命令，适合测试一次性脚本。
*   **系统环境**：如果你确有必要，可以使用 `uv pip install --system` 来安装包到系统 Python 环境，但这通常不推荐。

总而言之，`uv` 通过将虚拟环境管理深度集成到其工作流中，简化了 Python 项目的环境管理。你不再需要记忆 `activate` 和 `deactivate` 等命令，`uv` 会自动为你处理这一切。

更详细的说明和指南，可以查阅官方文档：
*   [uv 官方文档 (英文)](https://docs.astral.sh/uv/)
*   [uv 官方文档 (中文)](https://docs.astral.org.cn/uv/)