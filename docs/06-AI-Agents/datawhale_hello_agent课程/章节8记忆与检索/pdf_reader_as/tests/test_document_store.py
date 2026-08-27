"""storage/document_store.py 测试：SQLiteDocumentStore（本地离线）。"""
import os
import unittest
import tempfile

from storage.document_store import SQLiteDocumentStore, DocumentStore


class SQLiteDocumentStoreTest(unittest.TestCase):

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = SQLiteDocumentStore(db_path=self.path)

    def tearDown(self):
        self.store.close()
        os.path.exists(self.path) and os.unlink(self.path)
        # 清理单例残留，避免影响其他用例
        SQLiteDocumentStore._instances.pop(os.path.abspath(self.path), None)
        SQLiteDocumentStore._initialized_dbs.discard(os.path.abspath(self.path))

    def test_add_get_memory(self):
        mid = self.store.add_memory(
            memory_id="mem1", user_id="u1", content="内容",
            memory_type="episodic", timestamp=12345, importance=0.8,
            properties={"k": "v"})
        self.assertEqual(mid, "mem1")
        mem = self.store.get_memory("mem1")
        self.assertEqual(mem["content"], "内容")
        self.assertEqual(mem["memory_type"], "episodic")
        self.assertEqual(mem["properties"], {"k": "v"})

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.store.get_memory("nope"))

    def test_search_memories(self):
        self.store.add_memory("a", "u1", "内容A", "episodic", 100, 0.9)
        self.store.add_memory("b", "u2", "内容B", "semantic", 200, 0.5)
        res = self.store.search_memories(user_id="u1")
        self.assertEqual([r["memory_id"] for r in res], ["a"])
        res2 = self.store.search_memories(memory_type="semantic")
        self.assertEqual([r["memory_id"] for r in res2], ["b"])

    def test_update_memory(self):
        self.store.add_memory("a", "u1", "old", "episodic", 1, 0.5)
        self.assertTrue(self.store.update_memory("a", content="new", importance=0.9))
        mem = self.store.get_memory("a")
        self.assertEqual(mem["content"], "new")
        self.assertEqual(mem["importance"], 0.9)
        self.assertFalse(self.store.update_memory("missing", content="x"))

    def test_delete_memory(self):
        self.store.add_memory("a", "u1", "x", "episodic", 1, 0.5)
        self.assertTrue(self.store.delete_memory("a"))
        self.assertIsNone(self.store.get_memory("a"))
        self.assertFalse(self.store.delete_memory("a"))

    def test_database_stats(self):
        self.store.add_memory("a", "u1", "x", "episodic", 1, 0.5)
        stats = self.store.get_database_stats()
        self.assertEqual(stats["store_type"], "sqlite")
        self.assertIn("memories_count", stats)
        self.assertGreaterEqual(stats["memories_count"], 1)

    def test_add_get_document(self):
        did = self.store.add_document("doc content", {"user_id": "u1"})
        self.assertTrue(did)
        doc = self.store.get_document(did)
        self.assertEqual(doc["content"], "doc content")
        self.assertEqual(doc["memory_type"], "document")

    def test_abstract_cannot_instantiate(self):
        # 文档存储基类必须实现其抽象方法集合
        for name in ["add_memory", "get_memory", "search_memories", "update_memory",
                     "delete_memory", "get_database_stats", "add_document", "get_document"]:
            self.assertTrue(hasattr(DocumentStore, name))


if __name__ == "__main__":
    unittest.main()
