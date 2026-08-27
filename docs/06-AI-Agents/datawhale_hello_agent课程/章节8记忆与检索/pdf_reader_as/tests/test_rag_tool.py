"""rag_tool.py 测试（离线）：参数校验与预处理，不触发网络/Qdrant。"""
import unittest

from rag_tool import RAGTool


def _make_rag_tool():
    rag = RAGTool.__new__(RAGTool)
    rag.knowledge_base_path = "./knowledge_base"
    rag.collection_name = "rag_knowledge_base"
    rag.rag_namespace = "default"
    rag._pipelines = {}
    rag.initialized = True
    return rag


class RAGToolValidateTest(unittest.TestCase):

    def setUp(self):
        self.rag = _make_rag_tool()

    def test_valid_actions(self):
        cases = [
            {"action": "add_document", "file_path": "x"},
            {"action": "add_text", "text": "t"},
            {"action": "ask", "question": "q"},
            {"action": "search", "query": "q"},
            {"action": "stats"},
            {"action": "clear"},
        ]
        for params in cases:
            self.assertTrue(self.rag.validate_parameters(params), params)

    def test_missing_action_raises(self):
        with self.assertRaises(ValueError):
            self.rag.validate_parameters({})

    def test_add_document_requires_file(self):
        with self.assertRaises(ValueError):
            self.rag.validate_parameters({"action": "add_document"})

    def test_add_text_requires_text(self):
        with self.assertRaises(ValueError):
            self.rag.validate_parameters({"action": "add_text"})

    def test_ask_search_require_query(self):
        with self.assertRaises(ValueError):
            self.rag.validate_parameters({"action": "ask"})
        with self.assertRaises(ValueError):
            self.rag.validate_parameters({"action": "search"})

    def test_unsupported_action_passes_validation(self):
        # validate_parameters 不校验 action 名称合法性，交由 execute 处理
        self.assertTrue(self.rag.validate_parameters({"action": "boom"}))


class RAGToolPreprocessTest(unittest.TestCase):

    def setUp(self):
        self.rag = _make_rag_tool()

    def test_defaults_applied(self):
        out = self.rag._preprocess_parameters("search", query="q")
        self.assertEqual(out["limit"], 5)
        self.assertEqual(out["namespace"], "default")
        self.assertEqual(out["include_citations"], True)
        self.assertEqual(out["min_score"], 0.1)

    def test_user_values_preserved(self):
        out = self.rag._preprocess_parameters("search", query="q", limit=10, namespace="ns")
        self.assertEqual(out["limit"], 10)
        self.assertEqual(out["namespace"], "ns")


class RAGToolGetParametersTest(unittest.TestCase):

    def setUp(self):
        self.rag = _make_rag_tool()

    def test_parameters_include_action_and_content(self):
        names = {p.name for p in self.rag.get_parameters()}
        self.assertIn("action", names)
        self.assertIn("file_path", names)
        self.assertIn("text", names)
        self.assertIn("question", names)
        self.assertIn("query", names)


if __name__ == "__main__":
    unittest.main()
