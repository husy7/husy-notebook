"""embedding.py 工厂分派测试。

不加载真实模型，用 mock 替换各实现类，只验证分派/回退逻辑正确。
"""
import unittest
from unittest import mock

import embedding


class CreateEmbeddingModelDispatchTest(unittest.TestCase):

    @mock.patch("embedding.TFIDFEmbedding")
    def test_tfidf_dispatch(self, MockTFIDF):
        model = embedding.create_embedding_model("tfidf")
        self.assertIs(MockTFIDF.return_value, model)

    @mock.patch("embedding.LocalTransformerEmbedding")
    def test_local_dispatch(self, MockLocal):
        model = embedding.create_embedding_model("local", model_name="m")
        MockLocal.assert_called_once_with(model_name="m")
        self.assertIs(MockLocal.return_value, model)

    @mock.patch("embedding.DashScopeEmbedding")
    def test_dashscope_dispatch(self, MockDS):
        model = embedding.create_embedding_model("dashscope", model_name="m", api_key="k")
        MockDS.assert_called_once()
        self.assertIs(MockDS.return_value, model)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            embedding.create_embedding_model("bogus")


class CreateEmbeddingWithFallbackTest(unittest.TestCase):

    @mock.patch("embedding.create_embedding_model")
    def test_preferred_first(self, mock_create):
        # 首次（local）失败，第二次（dashscope）成功并返回
        mock_create.side_effect = [ValueError("first failed"), "success_model"]
        result = embedding.create_embedding_model_with_fallback(preferred_type="local")
        self.assertEqual(result, "success_model")
        calls = [c.args[0] for c in mock_create.call_args_list]
        self.assertEqual(calls, ["local", "dashscope"])


class GetDimensionTest(unittest.TestCase):

    @mock.patch("embedding.get_text_embedder")
    def test_uses_embedder_dimension(self, mock_embedder):
        mock_embedder.return_value.dimension = 512
        self.assertEqual(embedding.get_dimension(384), 512)

    @mock.patch("embedding.get_text_embedder")
    def test_fallback_on_error(self, mock_embedder):
        mock_embedder.side_effect = Exception("boom")
        self.assertEqual(embedding.get_dimension(384), 384)


if __name__ == "__main__":
    unittest.main()
