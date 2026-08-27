"""core/exceptions.py 测试：异常体系继承关系。"""
import unittest

from core.exceptions import (
    HelloAgentsException,
    LLMException,
    AgentException,
    ConfigException,
    ToolException,
)


class ExceptionsTest(unittest.TestCase):

    def test_hello_agents_is_base(self):
        self.assertTrue(issubclass(LLMException, HelloAgentsException))
        self.assertTrue(issubclass(AgentException, HelloAgentsException))
        self.assertTrue(issubclass(ConfigException, HelloAgentsException))
        self.assertTrue(issubclass(ToolException, HelloAgentsException))

    def test_raise_and_message(self):
        e = HelloAgentsException("boom")
        self.assertEqual(str(e), "boom")
        with self.assertRaises(HelloAgentsException):
            raise HelloAgentsException("fail")


if __name__ == "__main__":
    unittest.main()
