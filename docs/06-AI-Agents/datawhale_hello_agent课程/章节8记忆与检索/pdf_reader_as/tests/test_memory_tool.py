"""memory_tool.py 测试：MemoryTool 参数校验 + 动作分发（工作记忆，离线）。"""
import unittest

from memory_tool import MemoryTool


class MemoryToolValidateTest(unittest.TestCase):

    def setUp(self):
        self.tool = MemoryTool(user_id="tester", memory_types=["working"])

    def test_valid_actions_pass(self):
        for action in ["add", "search", "summary", "stats",
                       "update", "remove", "forget", "consolidate", "clear_all"]:
            params = {"action": action}
            if action in ("add",):
                params["content"] = "x"
            if action in ("search",):
                params["query"] = "x"
            if action in ("update", "remove"):
                params["memory_id"] = "x"
            self.assertTrue(self.tool.validate_parameters(params), action)

    def test_invalid_action_raises(self):
        with self.assertRaises(ValueError):
            self.tool.validate_parameters({"action": "delete"})  # 历史遗留，白名单已移除

    def test_add_requires_content(self):
        with self.assertRaises(ValueError):
            self.tool.validate_parameters({"action": "add"})

    def test_search_requires_query(self):
        with self.assertRaises(ValueError):
            self.tool.validate_parameters({"action": "search"})

    def test_update_remove_require_id(self):
        with self.assertRaises(ValueError):
            self.tool.validate_parameters({"action": "update"})
        with self.assertRaises(ValueError):
            self.tool.validate_parameters({"action": "remove"})

    def test_missing_action_raises(self):
        with self.assertRaises(ValueError):
            self.tool.validate_parameters({})


class MemoryToolActionTest(unittest.TestCase):

    def setUp(self):
        self.tool = MemoryTool(user_id="tester", memory_types=["working"])

    def test_add_search_flow(self):
        r = self.tool.run({"action": "add", "content": "机器学习定义", "importance": 0.9})
        self.assertIn("✅", r)
        s = self.tool.run({"action": "search", "query": "机器学习"})
        self.assertIn("🔍", s)

    def test_stats_and_summary(self):
        self.tool.run({"action": "add", "content": "记忆A", "importance": 0.8})
        self.assertIn("📈", self.tool.run({"action": "stats"}))
        self.assertIn("📊", self.tool.run({"action": "summary"}))

    def test_update_remove(self):
        self.tool.run({"action": "add", "content": "old"})
        # 从 memory_manager 拿到完整记忆 ID（run() 返回的只是前8位缩写）
        wm = self.tool.memory_manager.memory_types["working"]
        all_items = wm.get_all()
        self.assertEqual(len(all_items), 1)
        mem_id = all_items[0].id
        up = self.tool.run({"action": "update", "memory_id": mem_id, "content": "new"})
        self.assertIn("✅", up)
        rm = self.tool.run({"action": "remove", "memory_id": mem_id})
        self.assertIn("✅", rm)
        self.assertEqual(wm.get_stats()["count"], 0)

    def test_clear_all(self):
        self.tool.run({"action": "add", "content": "x"})
        self.assertIn("🧽", self.tool.run({"action": "clear_all"}))

    def test_right_flow(self):
        # consolidate 返回整合数量；clear_session 不抛异常
        out = self.tool.run({"action": "consolidate"})
        self.assertIn("🔄", out)
        self.tool.clear_session()

    def test_convenience_methods(self):
        self.tool.add_knowledge("知识要点", importance=0.9)
        ctx = self.tool.get_context_for_query("知识", limit=3)
        self.assertIsInstance(ctx, str)
        self.tool.clear_session()


if __name__ == "__main__":
    unittest.main()
