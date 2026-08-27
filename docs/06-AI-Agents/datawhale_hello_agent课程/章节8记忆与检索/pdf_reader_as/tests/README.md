# 功能测试

覆盖 `pdf_reader_agent` 各功能模块的测试代码。全部基于**标准库 `unittest`**，零第三方依赖，用项目 **agent 环境**的 python 运行。

## 运行方式

在项目根目录执行（务必用 agent 环境的 python）：

```bash
# Windows / PowerShell
& "E:\Anaconda3\envs\agent\python.exe" tests\run_all.py

# 详细输出
& "E:\Anaconda3\envs\agent\python.exe" tests\run_all.py -v

# 另：单项测试
& "E:\Anaconda3\envs\agent\python.exe" -m unittest tests.test_base
```

## 可选：集成测试（真实 Qdrant / LLM）

默认**跳过**，仅当显式开启 `RUN_INTEGRATION=1` 时执行。集成测试使用**独立临时 collection** 并会删除，避免污染你的真实数据：

```bash
$env:RUN_INTEGRATION="1"; & "E:\Anaconda3\envs\agent\python.exe" tests\run_all.py
```

> 集成测试会真实连接 Qdrant 云（读取 `.env` 的 `QDRANT_URL/API_KEY`）与本地 Ollama，耗时较长且依赖外部服务可用。

## 覆盖范围

| 文件 | 覆盖模块 |
| --- | --- |
| `test_base.py` | `Tool` / `ToolParameter` 基类模板方法 |
| `test_message.py` | `core.message.Message` |
| `test_config.py` | `core.config.Config` |
| `test_exceptions.py` | `core.exceptions` 异常体系 |
| `test_document.py` | `rag.document`：Document / DocumentProcessor / 加载 |
| `test_pipeline_chunking.py` | `rag.pipeline`：分块 / 去重 / 文本预处理 / 片段合并 |
| `test_embedding.py` | `embedding`：工厂分派 / 维度（mock，不加载模型） |
| `test_llm.py` | `core.llm`：provider 检测 / 凭证解析 / 调用（mock 网络）；含"纯净化调用"断言 |
| `test_working_memory.py` | `memory_types.working`：工作记忆 CRUD / 容量 / 遗忘 |
| `test_memory_manager.py` | `memory_manmager.MemoryManager`（工作记忆离线部分） |
| `test_memory_tool.py` | `memory_tool`：动作白名单 / 分发 / 便捷方法 |
| `test_document_store.py` | `storage.SQLiteDocumentStore` |
| `test_qdrant_store.py` | `storage.qdrant_store`：增删 / 检索 / 信息（mock 客户端） |
| `test_rag_tool.py` | `rag_tool`：参数校验 / 预处理 / 参数定义 |
| `test_integration.py` | 真实 Qdrant / LLM 链路（`RUN_INTEGRATION=1`） |

## 说明

- 纯逻辑测试在离线环境下直接可跑，不连接 Qdrant / Neo4j / LLM。
- 涉及网络/大型外部依赖的调用均被 mock 或标记为集成测试，避免测试污染真实数据与拖慢速度。
