# Smart Document Q&A Assistant · 智能文档问答助手

基于 **HelloAgents 框架** 的 PDF 智能文档问答 + 学习记忆助手。上传 PDF 构建向量知识库（RAG），进行智能问答，并记录学习历史（Memory），支持学习回顾与报告生成。

- **Web UI**：Gradio（`http://127.0.0.1:7860`）
- **知识库**：Qdrant（云端或本地）
- **LLM**：OpenAI 兼容接口（默认本地 Ollama）
- **记忆系统**：工作 / 情景 / 语义三类记忆 + 整合 / 遗忘

---

## ✨ 功能要点

- **智能问答直接查询已有知识库**：`ask()` 直接基于 Qdrant 已索引内容回答，无需先加载文档；检索不到才给出友好提示。
- **启动即用**：页面加载时**自动初始化助手**（默认 `web_user`）；"初始化助手"按钮仍可切换用户。
- **流式问答**：回答逐 token 浮现，首字即可见。
- **按需高级检索（MQE/HyDE）**：默认走秒级基础检索，结果不足或质量差时才启用 MQE+HyDE 重试。
- **多文档并行加载** + **`load_pdf` 后台化**：多 PDF 线程池并行解析入库，上传时界面即时反馈。
- **记忆全链路打通**：`add / search / summary / stats / update / remove / forget / consolidate / clear_all` 均可经 `run()` 调用。
- **LLM 纯净化调用**：仅做模型调用并接收返回，不干预模型的上下文/缓存/思考模式。

---

## 🚀 快速开始

### 1. 环境准备
```bash
conda env create -n agent python=3.10
conda activate agent
pip install -r requirements.txt
```

> **重要**：Windows 下 `conda activate agent` 在全新 PowerShell 会话可能落到 base（`E:\Anaconda3\python.exe`），而 base 环境缺少嵌入/UI 依赖会导致功能异常。**建议显式使用 agent 环境的解释器**：
> ```bash
> & "E:\Anaconda3\envs\agent\python.exe" main_.py
> ```

### 2. 配置
```bash
copy .env.example .env    # Windows
```
编辑 `.env`，填入 Qdrant、Ollama/LLM、嵌入等配置（详见 `.env.example` 内注释）。

### 3. 启动
```bash
python main_.py                     # 浏览器打开 http://127.0.0.1:7860
# 或显式用 agent 解释器：
& "E:\Anaconda3\envs\agent\python.exe" main_.py
```
Ollama 需已启动并拉取所需模型（如 `granite4.2:3b`），且 `LLM_BASE_URL` 指向 `http://localhost:11434/v1`。

---

## 🧪 测试

项目使用**标准库 `unittest`**（零第三方依赖）的功能测试，位于 `tests/`，覆盖框架、记忆、RAG、存储、嵌入、LLM 等全部功能：

```bash
# 运行全部离线单测（默认跳过需要真实 Qdrant/LLM 的集成测试）
& "E:\Anaconda3\envs\agent\python.exe" tests\run_all.py

# 额外运行真实 Qdrant / LLM 集成测试（使用独立临时 collection，用后清理）
$env:RUN_INTEGRATION="1"; & "E:\Anaconda3\envs\agent\python.exe" tests\run_all.py
```

详见 `tests/README.md`。

---

## 🗂️ 项目结构

```
main_.py            # 入口：PDFLearningAssistant + Gradio UI（自动初始化）
rag_tool.py         # RAG 工具（检索/问答/附件，含按需检索、流式）
memory_tool.py      # 记忆工具（add/search/summary/consolidate/forget 等）
memory_manmager.py  # 记忆管理器（协调记忆类型；拼写为历史遗留，勿改）
rag/                # RAG 管道（文档解析、分块、嵌入、向量检索）
storage/            # 存储层（Qdrant、Neo4j、SQLite 文档存储）
memory_types/       # 记忆类型实现（working/episodic/semantic/perceptual）
core/               # 框架（Agent / LLM / Config / Message）
embedding.py        # 统一嵌入模块（dashscope / local / tfidf 回退）
tests/              # 功能测试（标准库 unittest）
requirements.txt    # Python 依赖
CHANGELOG.md        # 全部改动与验证记录
```

---

## ⚙️ 环境变量一览（`.env.example`）

| 变量 | 说明 |
|------|------|
| `LLM_BASE_URL` / `LLM_MODEL_ID` | LLM 服务地址、模型名（本地 Ollama 或云端） |
| `LLM_API_KEY` | LLM API Key（Ollama 本地可留任意值） |
| `LLM_THINK` / `LLM_NUM_CTX` | 思考开关与上下文窗口（当前默认纯净化调用，不注入这些参数） |
| `LLM_TIMEOUT` | 请求超时秒数 |
| `QDRANT_URL` / `QDRANT_API_KEY` / `QDRANT_COLLECTION` | 向量数据库配置 |
| `RAG_NAMESPACE` | RAG 命名空间（默认 `default`；按用户隔离可设为自定义值） |
| `EMBED_MODEL_TYPE` / `EMBED_MODEL_NAME` | 嵌入提供方与模型 |
| `NEO4J_*` | 图数据库（可选，不可用自动降级为向量-only） |

---

## 📄 说明

本项目用于教学/学习（datawhale hello-agent-chat8 重构示例）。仓库已清理冗余/未接线文件；`abc.py` 为项目运行所需（`base.py` 从本地副本导入 `ABC`），保留。
