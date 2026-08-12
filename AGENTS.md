# AGENTS.md

> AI 助手在本仓库工作的必读指令。项目愿景、快速开始、部署等见 `README.md`。

## 当前状态

- 已 `git init`，处于本地开发阶段（无远程仓库）。
- 仓库现有文件：`AGENTS.md`、`README.md`、`.gitignore`、`docs/`（含目录骨架与 `docs/_template.md`）。
- `mkdocs.yml` 与 `.github/` 尚未创建——规划见 `README.md` 的 Roadmap。

## 技术栈与硬性规则

- 核心语言 **Python 3.10+**；所有代码片段**必须带类型注解**。
- 静态站点规划用 **MkDocs + Material 主题**（GitHub Pages，未配置）。
- `.codegraph/` 是本地代码索引数据，**切勿提交**（自带 `.gitignore`，已自动忽略）。
- `_template.md` 位于 **docs/ 根目录**，是新笔记的起点。
- AI 辅助生成的笔记，**必须在文末注明**"本文借助 AI 工具生成，经人工审核"。

## 笔记规范（要点）

- 结构：费曼五步法 = 入门（人话 + 类比）→ 深度拆解（机制 / 公式 / Mermaid）→ 手写可运行代码 → 权衡对比表 → 踩坑关联 → 费曼闭环（12 字总结 + 讲给小白听）。
- AI 应用类笔记额外加：实际输入 / 输出交互示例。
- 完整规范与使用指南见 `README.md`「如何写笔记」；模板本身见 `docs/_template.md`。
