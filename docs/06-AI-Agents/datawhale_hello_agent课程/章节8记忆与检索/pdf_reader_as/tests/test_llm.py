"""core/llm.py 逻辑测试（不触发真实 LLM 网络调用）。

覆盖：provider 自动检测、凭证解析、默认模型推断；以及 mock 后的流式/非流式调用。
"""
import os
import unittest
from unittest import mock

from core.llm import HelloAgentsLLM, HelloAgentsException


def _make_llm(**overrides):
    """构造 HelloAgentsLLM，但用 mock 替换 OpenAI 客户端，避免真实连接。"""
    with mock.patch("core.llm.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = mock.MagicMock()
        llm = HelloAgentsLLM(**overrides)
        return llm


class ProviderDetectionTest(unittest.TestCase):

    def setUp(self):
        self._old = {
            k: os.environ.get(k)
            for k in ["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL_ID",
                      "OLLAMA_HOST", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
                      "LLM_TIMEOUT", "LLM_THINK", "LLM_NUM_CTX"]
        }
        for k in self._old:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_deepseek_by_key(self):
        os.environ["DEEPSEEK_API_KEY"] = "ds-key"
        llm = _make_llm(api_key="x", base_url="http://localhost:11434/v1")
        # DEEPSEEK_API_KEY 优先 -> deepseek
        self.assertEqual(llm.provider, "deepseek")

    def test_ollama_by_localhost_port(self):
        llm = _make_llm(api_key="ollama", base_url="http://localhost:11434/v1", model="m")
        self.assertEqual(llm.provider, "ollama")

    def test_openai_base_url(self):
        llm = _make_llm(api_key="sk-abc", base_url="https://api.openai.com/v1", model="m")
        self.assertEqual(llm.provider, "openai")

    def test_localhost_else_local(self):
        llm = _make_llm(api_key="k", base_url="http://127.0.0.1:8080/v1", model="m")
        self.assertEqual(llm.provider, "local")


class CredentialResolveTest(unittest.TestCase):

    def test_ollama_credentials(self):
        os.environ["LLM_API_KEY"] = "mykey"
        llm = _make_llm(api_key="mykey", base_url="http://localhost:11434/v1", model="m")
        # 显式传入的 api_key 优先于默认 'ollama'
        self.assertEqual(llm.api_key, "mykey")

    def test_missing_creds_raises(self):
        os.environ.pop("LLM_API_KEY", None)
        os.environ.pop("LLM_BASE_URL", None)
        with mock.patch("core.llm.OpenAI"):
            with self.assertRaises(HelloAgentsException):
                HelloAgentsLLM(api_key="", base_url="")


class DefaultModelTest(unittest.TestCase):

    def test_ollama_default_model(self):
        os.environ["OLLAMA_HOST"] = "http://localhost:11434"
        os.environ["LLM_API_KEY"] = "ollama"
        os.environ["LLM_BASE_URL"] = "http://localhost:11434/v1"
        os.environ.pop("LLM_MODEL_ID", None)
        with mock.patch("core.llm.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = mock.MagicMock()
            llm = HelloAgentsLLM()
        self.assertEqual(llm.provider, "ollama")
        self.assertIn(llm.model, ("llama3.2", "granite4.2:3b"))


class InvokePureTest(unittest.TestCase):

    def test_invoke_returns_content(self):
        with mock.patch("core.llm.OpenAI") as MockOpenAI:
            client = mock.MagicMock()
            MockOpenAI.return_value = client
            client.chat.completions.create.return_value = mock.MagicMock(
                choices=[mock.MagicMock(message=mock.MagicMock(content="answer"))]
            )
            llm = HelloAgentsLLM(api_key="k", base_url="http://localhost:11434/v1", model="m")
            self.assertEqual(llm.invoke([{"role": "user", "content": "q"}]), "answer")

    def test_invoke_does_not_inject_think_numctx(self):
        """纯净化：不自动注入 think / num_ctx。"""
        with mock.patch("core.llm.OpenAI") as MockOpenAI:
            client = mock.MagicMock()
            MockOpenAI.return_value = client
            client.chat.completions.create.return_value = mock.MagicMock(
                choices=[mock.MagicMock(message=mock.MagicMock(content="ok"))]
            )
            llm = HelloAgentsLLM(api_key="k", base_url="http://localhost:11434/v1", model="m")
            llm.invoke([{"role": "user", "content": "q"}])
            _, kw = client.chat.completions.create.call_args
            # extra_body 不自动注入 think/num_ctx
            self.assertNotIn("num_ctx", kw.get("extra_body", {}))
            body = kw.get("extra_body") or {}
            self.assertNotIn("think", body)

    def test_stream_invoke_yields_chunks(self):
        """修复后的 think 方法可调用（不再 bool not callable）并逐块返回。"""
        with mock.patch("core.llm.OpenAI") as MockOpenAI:
            client = mock.MagicMock()
            MockOpenAI.return_value = client
            chunk = mock.MagicMock()
            chunk.choices = [mock.MagicMock(delta=mock.MagicMock(content="p"))]
            client.chat.completions.create.return_value = [chunk]
            llm = HelloAgentsLLM(api_key="k", base_url="http://localhost:11434/v1", model="m")
            parts = list(llm.stream_invoke([{"role": "user", "content": "q"}]))
            self.assertEqual(parts, ["p"])


if __name__ == "__main__":
    unittest.main()
