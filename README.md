# Aknowledge-base

> 面向 AI 应用方向的计算机知识库。核心语言 Python 3.9+，计划通过 MkDocs + Material 构建为静态网站（部署到 GitHub Pages）。

---

## 项目愿景

构建一个**面向 AI 应用方向的计算机知识库**，用于：

- 沉淀机器学习 / 深度学习理论与工程实践；
- 系统整理 **AI Agent / LLM 应用** 开发全链路（提示工程、RAG、工具调用、Agent 架构）；
- 覆盖传统 ML、NLP、CV 等领域的经典模型与最新进展；
- 记录 Python 工程化最佳实践（类型注解、异步、包管理、性能优化）；
- 积累面试必备的算法与数据结构（Python 实现）及计算机基础；
- 提供真实踩坑记录与解决方案。

**目标受众**：未来的自己（快速回顾）、潜在面试官或同行（展示技术深度）。
**最终形态**：通过 GitHub Pages 自动构建为静态网站，支持 Python 代码高亮、Mermaid 图表、MathJax 公式。

---

## 快速开始

> ⚠️ 项目处于早期阶段，部分能力尚未落地（见 [Roadmap](#roadmap)）。

1. **克隆仓库**（若尚未操作）：
   ```bash
   git clone <repo-url>
   cd knowledge-base
   ```
2. **安装 MkDocs 材料主题**（用于本地预览）：
   ```bash
   pip install mkdocs-material
   ```
3. **本地预览**（`mkdocs.yml` 配置完成后）：
   ```bash
   mkdocs serve
   ```
4. **编写笔记**：复制 `docs/_template.md` 为新笔记，遵循模板五段式结构。
5. **提交变更**：按下方「工作流程」提交。

---

## 目录结构（规划中，尚未创建）

> 以下目录骨架已创建（占位页 + `.gitkeep`），内容逐步填充中。

```
knowledge-base/
├── README.md
├── mkdocs.yml                      # 已创建
├── _template.md                    # 已迁入 docs/
├── docs/
│   ├── 00-Index/                   # MOC 知识地图（入口索引）
│   │   ├── MOC-Python.md
│   │   ├── MOC-ML-Algorithms.md
│   │   ├── MOC-DeepLearning.md
│   │   ├── MOC-NLP-LLM.md
│   │   ├── MOC-CV.md
│   │   ├── MOC-AI-Agents.md
│   │   ├── MOC-MLOps.md
│   │   └── MOC-CS-Basics.md
│   ├── 01-Python/                  # 语言特性、并发、打包、类型注解
│   ├── 02-ML-Algorithms/           # 线性模型、树模型、聚类、评估
│   ├── 03-DeepLearning/            # 基础、PyTorch、Model-Zoo
│   ├── 04-NLP-LLM/                 # 预处理、Embedding、Seq2Seq、LLM
│   ├── 05-CV/                      # 图像处理、CNN、目标检测
│   ├── 06-AI-Agents/               # Agent 架构、RAG、Prompt、框架、用例
│   ├── 07-MLOps-Deployment/        # ONNX、Triton、FastAPI、Docker-K8s
│   ├── 08-Algorithms-DSA/          # 排序、搜索、DP、图
│   ├── 09-CS-Foundation/           # 网络、OS、数据库
│   └── 10-Debug-Log/               # 踩坑记录（AI 方向尤其重要）
└── .github/
    └── workflows/
        └── deploy.yml              # GitHub Actions 部署到 Pages
```

---

## 如何写笔记

每篇笔记使用 `docs/_template.md` 作为起点，遵循**五段式结构**（认知构架 → 实践验证 → 全景视角 → 费曼闭环 → 执行意图）：

| 部分 | 强制程度 | 写作心法 |
| :--- | :--- | :--- |
| **1. 认知构架** | 🔴 强制 | 一句话直觉 + 心智图（双重编码）+ 第一性原理。先凭记忆写，写不出说明还没学懂，回去重看资料。 |
| **2. 实践验证** | 🔴 强制 | 手写最简实现 + 破坏性测试 + 变体对比表，通过"构造 / 破坏 / 对比"验证理解。 |
| **3. 全景视角** | 🟡 建议 | 方案选型矩阵 + 边界条件 + 关联知识网，明确"何时用、何时不用"。 |
| **4. 费曼闭环** | 🔴 强制 | 12 字总结 + 讲给 5 岁孩子 + Anki 卡片 + 倒推复原。能随时"提取"才算掌握。 |
| **5. 执行意图** | 🟡 建议 | If-Then 预设"场景 → 行动"，让知识在真实项目中自动触发。 |

**笔记链接约定**：笔记间使用标准 Markdown 链接（`[标题](路径)`），不使用 `[[双链]]` 语法。

**目录适配建议**：
- **纯算法笔记**（如快排）：忽略 AI 类坑点，专注边界条件；对比其他算法（如归并排序）。
- **AI 应用笔记**（如 Agent）：代码示例增加输入 / 输出日志打印，体现交互过程；对比表加"Token 成本"维度。

**其他要求**：
- 所有 Python 代码带类型注解，使用 Python 3.10+。
- AI 辅助生成的笔记，文末注明"本文借助 AI 工具生成，经人工审核"。

---

## 工作流程

- **Git 初始化**：已执行 `git init`。`.gitignore` 忽略：
  ```
  .codegraph/
  .omo/
  .reasonix/
  __pycache__/
  *.pyc
  .venv/
  site/
  .DS_Store
  ```
- **分支策略**：
  - `main`：稳定版，仅接受 PR 合并。
  - `dev`：日常开发分支。
  - 新功能 / 新笔记：从 `dev` 切出 `feature/<topic>`，完成后 PR 到 `dev`。
- **提交信息前缀**：`docs:` 笔记 / `code:` 可运行脚本 / `fix:` 修复 / `refactor:` 重构 / `ci:` CI 配置。
- **审核**：合并前进行代码片段语法检查（`mypy` 或 `pytest` 简单验证）。

---

## 部署（GitHub Pages，规划中）

- 使用 GitHub Actions 自动构建，配置文件 `.github/workflows/deploy.yml`（待创建）。
- 构建命令：`mkdocs build --clean`，将 `site/` 部署到 `gh-pages` 分支。
- 首次部署需在仓库 Settings > Pages 中启用 "Deploy from branch" 并选择 `gh-pages`。

---

## Roadmap

- [x] 执行 `git init`，配置 `.gitignore`
- [x] 创建 `docs/` 目录结构
- [x] 配置 `mkdocs.yml` 并测试本地构建
- [ ] 编写首批核心笔记（按优先级）：
  1. ✅ `01-Python/Language-Features/Decorators.md`（示例笔记，已完成）
  2. `03-DeepLearning/PyTorch/Tensor-Basics.md`
  3. `02-ML-Algorithms/Linear-Models/LinearRegression-Code.md`
  4. `06-AI-Agents/Agent-Architecture/What-is-Agent.md`
- [ ] 提交初始 PR，启用 GitHub Actions
- [ ] 持续填充内容，每周至少 1~2 篇新笔记

---

## 贡献与维护

- 定期更新 MOC（内容地图），确保所有笔记被索引。
- 每条笔记末尾添加"相关 MOC"链接，形成网状结构。
- `.codegraph/` 为本地代码索引工具数据，切勿提交。
