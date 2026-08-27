"""base.py 框架层测试：ToolParameter 与 Tool 基类。"""
import unittest

from base import Tool, ToolParameter


class DummyTool(Tool):
    """最小可实例化 Tool 子类，用于测试基类模板方法。"""

    def __init__(self):
        super().__init__(name="dummy", description="dummy tool")

    def get_parameters(self):
        return [
            ToolParameter(name="required_flag", type="string", description="必需参数", required=True),
            ToolParameter(name="optional_flag", type="string", description="可选参数", required=False, default="x"),
        ]

    def _execute(self, parameters):
        return f"exec:{parameters.get('required_flag')}"


class ToolParameterTest(unittest.TestCase):

    def test_defaults(self):
        p = ToolParameter(name="a", type="string", description="d")
        self.assertEqual(p.required, True)
        self.assertIsNone(p.default)

    def test_schema(self):
        p = ToolParameter(name="a", type="integer", description="d", required=False)
        self.assertEqual(p.schema, {"name": "a", "type": "integer", "required": False})


class ToolBaseTest(unittest.TestCase):

    def setUp(self):
        self.tool = DummyTool()

    def test_run_success(self):
        out = self.tool.run({"required_flag": "hello"})
        self.assertEqual(out, "exec:hello")

    def test_run_missing_required(self):
        # 缺必需参数时返回中文错误提示
        self.assertEqual(self.tool.run({}), "❌ 参数验证失败：缺少必需的参数")

    def test_validate_parameters(self):
        self.assertTrue(self.tool.validate_parameters({"required_flag": "1"}))
        self.assertFalse(self.tool.validate_parameters({}))

    def test_to_dict(self):
        d = self.tool.to_dict()
        self.assertEqual(d["name"], "dummy")
        self.assertEqual(len(d["parameters"]), 2)

    def test_to_str(self):
        self.assertIn("dummy", self.tool.to_str())

    def test_get_parameters_types(self):
        for p in self.tool.get_parameters():
            self.assertIsInstance(p, ToolParameter)


if __name__ == "__main__":
    unittest.main()
