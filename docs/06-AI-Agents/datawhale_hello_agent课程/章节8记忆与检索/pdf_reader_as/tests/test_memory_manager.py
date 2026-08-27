"""memory_manmager.py 测试：MemoryManager 离线部分（仅工作记忆）。

不启用 episodic/semantic，避免连接 Qdrant/Neo4j。
"""
import unittest

from memory_manmager import MemoryManager, MemoryConfig


class MemoryManagerWorkingOnlyTest(unittest.TestCase):

    def setUp(self):
        self.mm = MemoryManager(
            user_id="tester",
            enable_working=True,
            enable_episodic=False,
            enable_semantic=False,
        )

    def test_add_and_retrieve(self):
        # 显式 auto_classify=False，避免内容含"原理"被自动归类为未启用的 semantic
        mid = self.mm.add_memory(content="机器学习的原理", memory_type="working",
                                 importance=0.9, auto_classify=False)
        self.assertTrue(mid)
        results = self.mm.retrieve_memories("机器学习", limit=5)
        self.assertGreaterEqual(len(results), 1)

    def test_add_with_auto_classify(self):
        # auto_classify 会按关键词分类；这里无关键词 -> working
        mid = self.mm.add_memory(content="这是普普通通的内容", auto_classify=True)
        self.assertTrue(mid)

    def test_update_remove(self):
        mid = self.mm.add_memory(content="旧内容", memory_type="working")
        self.assertTrue(self.mm.update_memory(mid, content="新内容"))
        self.assertTrue(self.mm.remove_memory(mid))
        self.assertFalse(self.mm.remove_memory("不存在"))

    def test_get_memory_stats(self):
        self.mm.add_memory(content="第一条重要记忆", memory_type="working", importance=0.8)
        stats = self.mm.get_memory_stats()
        self.assertEqual(stats["user_id"], "tester")
        self.assertIn("working", stats["enabled_types"])
        self.assertGreaterEqual(stats["total_memories"], 1)

    def test_forget_and_clear(self):
        self.mm.add_memory(content="不重要", memory_type="working", importance=0.05)
        n = self.mm.forget_memories(strategy="importance_based", threshold=0.1)
        self.assertIsInstance(n, int)
        self.mm.clear_all_memories()
        self.assertEqual(self.mm.get_memory_stats()["total_memories"], 0)

    def test_unsupported_type_raises(self):
        with self.assertRaises(ValueError):
            self.mm.add_memory(content="x", memory_type="nope", auto_classify=False)


class MemoryClassifierTest(unittest.TestCase):

    def setUp(self):
        self.mm = MemoryManager(
            enable_working=True, enable_episodic=False, enable_semantic=False)

    def test_episodic_classification(self):
        # 含 "昨天" -> episodic
        self.assertEqual(self.mm._classify_memory_type("昨天发生了一件事", None), "episodic")

    def test_semantic_classification(self):
        # 含 "定义" -> semantic
        self.assertEqual(self.mm._classify_memory_type("给出定义内容", None), "semantic")

    def test_importance_calculation(self):
        # 长度>100 加0.1、含关键词"关键"加0.2 -> 0.5+0.1+0.2=0.8
        h = self.mm._calculate_importance("关键必项很重要" * 20, None)
        self.assertAlmostEqual(h, 0.8)
        low = self.mm._calculate_importance("短", None)
        self.assertAlmostEqual(low, 0.5)


if __name__ == "__main__":
    unittest.main()
