"""core/message.py 测试：Message 数据类。"""
import unittest
from datetime import datetime

from core.message import Message


class MessageTest(unittest.TestCase):

    def test_creation_defaults(self):
        m = Message("hi", "user")
        self.assertEqual(m.content, "hi")
        self.assertEqual(m.role, "user")
        self.assertIsInstance(m.timestamp, datetime)
        self.assertEqual(m.metadata, {})

    def test_to_dict(self):
        m = Message("hello", "assistant")
        self.assertEqual(m.to_dict(), {"role": "assistant", "content": "hello"})

    def test_system_role(self):
        m = Message("sys", "system")
        self.assertEqual(m.role, "system")

    def test_str(self):
        m = Message("abc", "user")
        self.assertEqual(str(m), "[user] abc")


if __name__ == "__main__":
    unittest.main()
