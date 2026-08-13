---
title: "Python（01）"
description: "Python 语言特性、并发编程、包管理与类型注解——AI 应用的工程地基。"
tags: [Python]
---

# Python（01）

> **板块定位**：Python 是 AI 应用开发的核心语言。本板块沉淀语言特性、并发、打包与类型注解的最佳实践，所有代码遵循 Python 3.10+ 与类型注解规范。

## 覆盖主题

| 子目录 | 覆盖内容 | 状态 |
| :--- | :--- | :--- |
| `Language-Features` | 装饰器、生成器、闭包、上下文管理器、元类 | 🟡 已有示例笔记 |
| `Concurrency` | GIL、多线程、asyncio、进程池、锁 | 🔴 待填充 |
| `Packaging` | pyproject.toml、uv、虚拟环境、依赖管理 | 🔴 待填充 |
| `Type-Hints` | typing 进阶、泛型、Protocol、mypy/pyright | 🔴 待填充 |

## 规划笔记

- [x] [装饰器（示例笔记）](Language-Features/Decorators.md) —— 新模板五段式示范
- [ ] 闭包与作用域（Closures）
- [ ] 生成器与迭代器（Generators）
- [ ] asyncio 事件循环与协程
- [ ] 现代打包：pyproject.toml 实战
- [ ] 类型注解进阶：`ParamSpec` / `TypeVar` / `Protocol`

## 写作约定

- 所有代码片段**必须带类型注解**（Python 3.10+，优先使用 `X | None` 语法）。
- AI 辅助生成的笔记，文末注明"本文借助 AI 工具生成，经人工审核"。
