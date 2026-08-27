"""storage/qdrant_store.py 测试。

默认用 mock 客户端做单元测试（离线），不连真实 Qdrant。
集成（真实连接/CRUD）由 test_integration.py 在 RUN_INTEGRATION=1 时执行。
"""
import unittest
from unittest import mock

from storage.qdrant_store import QdrantVectorStore, QdrantConnectionManager


def _make_store(vector_size=384):
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.url = "http://fake"
    store.api_key = "k"
    store.collection_name = "test_coll"
    store.vector_size = vector_size
    store.timeout = 5
    store.search_ef = 128
    store.search_exact = False
    store.distance = mock.MagicMock()
    store.client = mock.MagicMock()
    return store


class AddVectorsTest(unittest.TestCase):

    def test_empty_returns_zero(self):
        store = _make_store()
        self.assertEqual(store.add_vectors([], []), 0)

    def test_dimension_mismatch_skips(self):
        store = _make_store(vector_size=384)
        store.client.upsert.side_effect = None
        n = store.add_vectors([[0.1] * 100], [{"a": 1}], ["id1"])
        self.assertEqual(n, 0)  # 维度不符被跳过

    def test_valid_float_ids(self):
        store = _make_store()
        n = store.add_vectors([[0.1] * 384], [{"a": 1}], [123])
        self.assertEqual(n, 1)
        store.client.upsert.assert_called_once()

    def test_invalid_str_id_regenerated_with_original(self):
        store = _make_store()
        n = store.add_vectors([[0.1] * 384], [{"a": 1}], ["not-a-uuid"])
        self.assertEqual(n, 1)
        call = store.client.upsert.call_args
        kwargs = call.kwargs if call.kwargs else call[1]
        points = kwargs["points"]
        self.assertIn("original_id", points[0].payload)
        self.assertEqual(points[0].payload["original_id"], "not-a-uuid")

    def test_timestamp_added(self):
        store = _make_store()
        store.add_vectors([[0.1] * 384], [{"a": 1}], ["11111111-1111-1111-1111-111111111111"])
        points = store.client.upsert.call_args.kwargs["points"]
        self.assertIn("timestamp", points[0].payload)
        self.assertIn("added_at", points[0].payload)


class SearchSimilarTest(unittest.TestCase):

    def test_wrong_dim_returns_empty(self):
        store = _make_store(vector_size=384)
        self.assertEqual(store.search_similar([0.1] * 10), [])

    def test_builds_filter_and_returns_results(self):
        store = _make_store()
        store.client.query_points.return_value = mock.MagicMock(points=[
            mock.MagicMock(id="p1", score=0.9, payload={"content": "x"}),
        ])
        res = store.search_similar([0.1] * 384, limit=3, where={"memory_type": "rag_chunk"})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "p1")
        call = store.client.query_points.call_args
        kwargs = call.kwargs if call.kwargs else call[1]
        self.assertEqual(kwargs["limit"], 3)
        # 过滤条件已构造
        self.assertIsNotNone(kwargs["query_filter"])

    def test_no_where_no_filter(self):
        store = _make_store()
        store.client.query_points.return_value = mock.MagicMock(points=[])
        store.search_similar([0.1] * 384, limit=5, where=None)
        kwargs = store.client.query_points.call_args.kwargs
        self.assertIsNone(kwargs["query_filter"])


class CollectionInfoTest(unittest.TestCase):

    def test_get_collection_info_maps_fields(self):
        store = _make_store()
        store.client.get_collection.return_value = mock.MagicMock(
            vectors_count=10, indexed_vectors_count=5, points_count=10, segments_count=2,
        )
        store.vector_size = 384
        store.distance.value = "Cosine"
        info = store.get_collection_info()
        self.assertEqual(info["vectors_count"], 10)
        self.assertEqual(info["points_count"], 10)
        self.assertEqual(info["config"]["vector_size"], 384)

    def test_health_check(self):
        store = _make_store()
        store.client.get_collections.return_value = None
        self.assertTrue(store.health_check())
        store.client.get_collections.side_effect = Exception("down")
        self.assertFalse(store.health_check())


class ConnectionManagerTest(unittest.TestCase):

    def test_get_instance_caches(self):
        with mock.patch.object(QdrantVectorStore, "__init__", lambda *a, **k: None):
            manager = QdrantConnectionManager
            old = dict(manager._instances)
            manager._instances.clear()
            try:
                i1 = manager.get_instance(url="http://u", api_key="k",
                                          collection_name="c1", vector_size=3)
                i2 = manager.get_instance(url="http://u", api_key="k",
                                          collection_name="c1", vector_size=3)
                self.assertIs(i1, i2)
            finally:
                manager._instances.clear()
                manager._instances.update(old)


if __name__ == "__main__":
    unittest.main()
