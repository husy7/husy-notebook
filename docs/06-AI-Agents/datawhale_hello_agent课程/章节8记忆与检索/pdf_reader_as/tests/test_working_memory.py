"""memory_types/working.py 测试：工作记忆（纯内存，离线可测）。"""
import unittest
from datetime import datetime

from memory_base import MemoryConfig, MemoryItem
from memory_types.working import WorkingMemory


def _item(content="记忆内容", importance=0.5, **kw):
    return MemoryItem(
        id=kw.get("id", "id_1"),
        content=content,
        memory_type="working",
        user_id=kw.get("user_id", "u1"),
        timestamp=kw.get("timestamp", datetime.now()),
        importance=importance,
        metadata=kw.get("metadata", {}),
    )


class WorkingMemoryTest(unittest.TestCase):

    def setUp(self):
        self.wm = WorkingMemory(MemoryConfig())

    def test_add_and_retrieve_exact(self):
        self.wm.add(_item(content="机器学习很关键", importance=0.9, id="m1"))
        results = self.wm.retrieve("机器学习", limit=5)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].id, "m1")

    def test_capacity_enforced(self):
        cfg = MemoryConfig()
        cfg.working_memory_capacity = 3
        wm = WorkingMemory(cfg)
        for i in range(10):
            wm.add(_item(content=f"记忆第{i}条", importance=0.5, id=f"id{i}"))
        stats = wm.get_stats()
        self.assertLessEqual(stats["count"], 3)

    def test_remove(self):
        self.wm.add(_item(id="m1"))
        self.assertTrue(self.wm.has_memory("m1"))
        self.assertTrue(self.wm.remove("m1"))
        self.assertFalse(self.wm.has_memory("m1"))
        self.assertFalse(self.wm.remove("m1"))

    def test_update(self):
        m = _item(id="m1", content="old", importance=0.4)
        self.wm.add(m)
        self.assertTrue(self.wm.update("m1", content="new", importance=0.9))
        results = self.wm.retrieve("new", limit=5)
        self.assertEqual(results[0].content, "new")
        self.assertFalse(self.wm.update("missing", content="x"))

    def test_clear(self):
        self.wm.add(_item(id="a"))
        self.wm.add(_item(id="b"))
        self.wm.clear()
        self.assertEqual(self.wm.get_stats()["count"], 0)

    def test_get_stats_fields(self):
        self.wm.add(_item(id="a", importance=0.7))
        s = self.wm.get_stats()
        self.assertEqual(s["memory_type"], "working")
        self.assertIn("count", s)
        self.assertIn("current_tokens", s)

    def test_get_all_and_recent(self):
        self.wm.add(_item(id="a", content="第一条"))
        self.wm.add(_item(id="b", content="第二条"))
        self.assertEqual(len(self.wm.get_all()), 2)
        self.assertGreaterEqual(len(self.wm.get_recent(1)), 1)

    def test_get_important(self):
        self.wm.add(_item(id="low", content="低", importance=0.1))
        self.wm.add(_item(id="high", content="高", importance=0.9))
        imp = self.wm.get_important(5)
        self.assertEqual(imp[0].id, "high")

    def test_forget_importance_based(self):
        self.wm.add(_item(id="keep", content="重要事项", importance=0.9))
        self.wm.add(_item(id="drop", content="不重要内容", importance=0.05))
        n = self.wm.forget("importance_based", threshold=0.1)
        self.assertGreaterEqual(n, 1)
        self.assertFalse(self.wm.has_memory("drop"))


if __name__ == "__main__":
    unittest.main()
