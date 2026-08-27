"""core/config.py 测试：Config 数据类。"""
import os
import unittest

from core.config import Config


class ConfigTest(unittest.TestCase):

    def test_defaults(self):
        c = Config()
        self.assertEqual(c.default_model, "gpt-3.5-turbo")
        self.assertEqual(c.default_provider, "openai")
        self.assertEqual(c.temperature, 0.7)
        self.assertIsNone(c.max_tokens)
        self.assertFalse(c.debug)
        self.assertEqual(c.max_history_length, 100)

    def test_custom_values(self):
        c = Config(default_model="llama", temperature=0.2, max_tokens=128)
        self.assertEqual(c.default_model, "llama")
        self.assertEqual(c.temperature, 0.2)
        self.assertEqual(c.max_tokens, 128)

    def test_from_env(self):
        old = {k: os.environ.get(k) for k in ("DEBUG", "TEMPERATURE", "MAX_TOKENS")}
        try:
            os.environ["DEBUG"] = "true"
            os.environ["TEMPERATURE"] = "0.1"
            os.environ["MAX_TOKENS"] = "256"
            c = Config.from_env()
            self.assertTrue(c.debug)
            self.assertAlmostEqual(c.temperature, 0.1)
            self.assertEqual(c.max_tokens, 256)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_to_dict(self):
        c = Config()
        d = c.to_dict()
        self.assertEqual(d["default_model"], "gpt-3.5-turbo")


if __name__ == "__main__":
    unittest.main()
