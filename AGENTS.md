# AGENTS.md

> AI 助手在本仓库工作的必读指令。项目愿景、快速开始、部署等见 `README.md`。

## 当前状态

- 已 `git init`（默认分支 `main`），处于本地开发阶段（无远程仓库）。
- 仓库现有文件：`AGENTS.md`、`README.md`、`.gitignore`、`docs/`（含目录骨架、`docs/_template.md` 与 `docs/javascripts/mathjax.js`）、`mkdocs.yml`（Material 主题 + Mermaid + MathJax，已通过 `mkdocs build` 验证）。
- `.github/` 尚未创建（无远程仓库，CI 部署规划见 `README.md` 的 Roadmap）。

## 技术栈与硬性规则

- 核心语言 **Python 3.10+**；所有代码片段**必须带类型注解**。
- 静态站点规划用 **MkDocs + Material 主题**（GitHub Pages，未配置）。
- `.codegraph/` 是本地代码索引数据，**切勿提交**（自带 `.gitignore`，已自动忽略）。
- `_template.md` 位于 **docs/ 根目录**，是新笔记的起点。
- AI 辅助生成的笔记，**必须在文末注明**"本文借助 AI 工具生成，经人工审核"。

## 笔记规范（要点）

- 结构：五段式 = 认知构架（直觉 + 心智图 + 第一性原理）→ 实践验证（手写 + 破坏性测试 + 变体对比 + 排雷）→ 全景视角（选型矩阵 + 边界条件 + 知识网络）→ 费曼闭环（12 字 + 讲给小孩 + Anki + 倒推复原）→ 执行意图（If-Then 触发）。
- 笔记间使用**标准 Markdown 链接**（`[标题](路径)`），不使用 `[[双链]]`。
- AI 应用类笔记额外加：实际输入 / 输出交互示例。
- 完整规范与使用指南见 `README.md`「如何写笔记」；模板本身见 `docs/_template.md`。
