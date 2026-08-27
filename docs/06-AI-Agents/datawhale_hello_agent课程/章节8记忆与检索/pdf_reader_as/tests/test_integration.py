"""集成测试：真实 Qdrant / LLM 调用。

默认跳过，仅当设置环境变量 RUN_INTEGRATION=1 时执行。
为避免污染真实数据：
- RAG 使用独立临时 collection；
- 记忆使用临时 QDRANT_COLLECTION 与临时 storage_path；
- 用后删除临时 collection。

运行示例：
    RUN_INTEGRATION=1 python -m unittest tests.test_integration
"""
import os
import tempfile
import unittest

RUN_INTEGRATION = os.getenv("RUN_INTEGRATION", "0") == "1"


@unittest.skipUnless(RUN_INTEGRATION, "设置 RUN_INTEGRATION=1 才运行集成测试")
class RAGToolIntegrationTest(unittest.TestCase):
    """RAG 增删查链路（使用独立临时 collection）。"""

    COLLECTION = "dsh_test_rag_coll"

    def setUp(self):
        from rag_tool import RAGTool
        from dotenv import load_dotenv
        load_dotenv()
        self.rag = RAGTool(
            collection_name=self.COLLECTION,
            rag_namespace="default",
            knowledge_base_path=tempfile.mkdtemp(),
        )

    def tearDown(self):
        # 清理临时集合，避免残留
        try:
            self.rag.run({"action": "clear", "confirm": True})
        except Exception:
            pass

    def test_add_text_and_search(self):
        r = self.rag.run({"action": "add_text", "text": "机器学习是现代人工智能的核心方法。", "chunk_size": 100})
        self.assertIn("✅", r)
        out = self.rag.run({"action": "search", "query": "机器学习", "enable_advanced_search": False})
        self.assertNotIn("未找到", out)
        self.assertIn("搜索结果", out)

    def test_stats(self):
        out = self.rag.run({"action": "stats"})
        self.assertIn("RAG 知识库统计", out)


@unittest.skipUnless(RUN_INTEGRATION, "设置 RUN_INTEGRATION=1 才运行集成测试")
class MemoryToolEpisodicIntegrationTest(unittest.TestCase):
    """情景/语义记忆（使用临时 QDRANT_COLLECTION 隔离）。"""

    def setUp(self):
        import os
        os.environ["QDRANT_COLLECTION"] = "dsh_test_mem_coll"
        from memory_tool import MemoryTool
        from memory_base import MemoryConfig
        from dotenv import load_dotenv
        load_dotenv()
        cfg = MemoryConfig(storage_path=tempfile.mkdtemp())
        self.tool = MemoryTool(user_id="tester", memory_config=cfg,
                               memory_types=["working", "episodic"])

    def test_episodic_add_search_flow(self):
        r = self.tool.run({"action": "add", "content": "今天学习了机器学习",
                           "memory_type": "episodic", "importance": 0.8})
        self.assertIn("✅", r)
        s = self.tool.run({"action": "search", "query": "机器学习", "memory_type": "episodic"})
        self.assertIn("🔍", s)


if __name__ == "__main__":
    unittest.main()
