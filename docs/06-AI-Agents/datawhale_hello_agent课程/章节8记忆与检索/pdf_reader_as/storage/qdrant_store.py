"""
Qdrant向量数据库存储实现（基于最新 qdrant-client API）
"""

import logging
import os
import uuid
import threading
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    # QueryVector 不需要导入，直接传向量列表
)
from dotenv import load_dotenv

load_dotenv()

# 仅在调试模式打印连接信息（不打印 API Key）
if os.getenv("DEBUG", "").lower() == "true":
    print("QDRANT_URL =", repr(os.getenv("QDRANT_URL")))

logger = logging.getLogger(__name__)


class QdrantConnectionManager:
    """Qdrant连接管理器 - 防止重复连接和初始化"""
    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "hello_agents_vectors",
        vector_size: int = 384,
        distance: str = "cosine",
        timeout: int = 30,
        **kwargs
    ) -> "QdrantVectorStore":
        key = (url if url is not None else "local", collection_name)
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    logger.debug(f"🔄 创建新的Qdrant连接: {collection_name}")
                    cls._instances[key] = QdrantVectorStore(
                        url=url,
                        api_key=api_key,
                        collection_name=collection_name,
                        vector_size=vector_size,
                        distance=distance,
                        timeout=timeout,
                        **kwargs
                    )
        return cls._instances[key]


class QdrantVectorStore:
    """Qdrant向量数据库存储实现"""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "hello_agents_vectors",
        vector_size: int = 384,
        distance: str = "cosine",
        timeout: int = 30,
        **kwargs
    ):
        self.url = url if url is not None else os.getenv("QDRANT_URL")
        self.api_key = api_key if api_key is not None else os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "hello_agents_vectors")
        self.vector_size = vector_size or int(os.getenv("QDRANT_VECTOR_SIZE", 384))
        self.timeout = timeout or int(os.getenv("QDRANT_TIMEOUT", 30))

        # HNSW / 搜索参数
        try:
            self.hnsw_m = int(os.getenv("QDRANT_HNSW_M", "32"))
        except Exception:
            self.hnsw_m = 32
        try:
            self.hnsw_ef_construct = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "256"))
        except Exception:
            self.hnsw_ef_construct = 256
        try:
            self.search_ef = int(os.getenv("QDRANT_SEARCH_EF", "128"))
        except Exception:
            self.search_ef = 128
        self.search_exact = os.getenv("QDRANT_SEARCH_EXACT", "0") == "1"

        distance_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID,
        }
        self.distance = distance_map.get(distance.lower(), Distance.COSINE)

        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        try:
            if self.url and self.api_key:
                self.client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=self.timeout
                )
                logger.info(f"✅ 连接Qdrant云服务: {self.url}")
            elif self.url:
                self.client = QdrantClient(
                    url=self.url,
                    timeout=self.timeout
                )
                logger.info(f"✅ 连接Qdrant服务: {self.url}")
            else:
                self.client = QdrantClient(
                    host="localhost",
                    port=6333,
                    timeout=self.timeout
                )
                logger.info("✅ 连接本地Qdrant: localhost:6333")

            # 记录版本
            import qdrant_client
            try:
                logger.info(f"📦 qdrant-client 版本: {qdrant_client.__version__}")
            except AttributeError:
                logger.info("📦 qdrant-client 版本: 未知")

            self._ensure_collection()

        except Exception as e:
            logger.error(f"❌ Qdrant连接失败: {e}")
            if not self.url:
                logger.info("💡 本地连接失败，可以考虑使用Qdrant云服务")
                logger.info("💡 或启动本地服务: docker run -p 6333:6333 qdrant/qdrant")
            raise

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            if self.collection_name not in [c.name for c in collections]:
                hnsw_cfg = None
                try:
                    hnsw_cfg = models.HnswConfigDiff(
                        m=self.hnsw_m,
                        ef_construct=self.hnsw_ef_construct
                    )
                except Exception:
                    hnsw_cfg = None
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    ),
                    hnsw_config=hnsw_cfg
                )
                logger.info(f"✅ 创建集合: {self.collection_name}")
            else:
                logger.info(f"✅ 使用现有集合: {self.collection_name}")
                try:
                    if hasattr(models, "HnswConfigDiff"):
                        self.client.update_collection(
                            collection_name=self.collection_name,
                            hnsw_config=models.HnswConfigDiff(
                                m=self.hnsw_m,
                                ef_construct=self.hnsw_ef_construct
                            )
                        )
                except Exception:
                    pass

            self._ensure_payload_indexes()

        except Exception as e:
            logger.error(f"❌ 集合初始化失败: {e}")
            raise

    def _ensure_payload_indexes(self):
        try:
            index_fields = [
                ("memory_type", models.PayloadSchemaType.KEYWORD),
                ("user_id", models.PayloadSchemaType.KEYWORD),
                ("memory_id", models.PayloadSchemaType.KEYWORD),
                ("original_id", models.PayloadSchemaType.KEYWORD),
                ("timestamp", models.PayloadSchemaType.INTEGER),
                ("modality", models.PayloadSchemaType.KEYWORD),
                ("source", models.PayloadSchemaType.KEYWORD),
                ("external", models.PayloadSchemaType.BOOL),
                ("namespace", models.PayloadSchemaType.KEYWORD),
                ("is_rag_data", models.PayloadSchemaType.BOOL),
                ("rag_namespace", models.PayloadSchemaType.KEYWORD),
                ("data_source", models.PayloadSchemaType.KEYWORD),
            ]
            for field_name, schema_type in index_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema_type,
                    )
                except Exception as ie:
                    if "already exists" in str(ie).lower():
                        logger.debug(f"索引 {field_name} 已存在")
                    else:
                        logger.warning(f"创建索引 {field_name} 失败: {ie}")
        except Exception as e:
            logger.debug(f"创建payload索引时出错: {e}")

    def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[Union[int, str]]] = None
    ) -> int:
        try:
            if not vectors:
                return 0

            if ids is None:
                ids = [f"vec_{i}_{int(datetime.now().timestamp() * 1000000)}"
                       for i in range(len(vectors))]

            points = []
            for i, (vector, meta, point_id) in enumerate(zip(vectors, metadata, ids)):
                if len(vector) != self.vector_size:
                    logger.warning(f"⚠️ 向量维度不匹配: 期望{self.vector_size}, 实际{len(vector)}")
                    continue

                # ID格式校验：Qdrant接受无符号整数或UUID字符串
                if isinstance(point_id, int) and point_id >= 0:
                    safe_id = point_id
                elif isinstance(point_id, str):
                    try:
                        uuid.UUID(point_id)
                        safe_id = point_id
                    except ValueError:
                        safe_id = str(uuid.uuid4())
                        meta = meta.copy()
                        meta["original_id"] = point_id
                else:
                    safe_id = str(uuid.uuid4())

                meta_with_timestamp = meta.copy()
                meta_with_timestamp["timestamp"] = int(datetime.now().timestamp())
                meta_with_timestamp["added_at"] = int(datetime.now().timestamp())

                if "external" in meta_with_timestamp and not isinstance(
                    meta_with_timestamp.get("external"), bool
                ):
                    val = meta_with_timestamp.get("external")
                    meta_with_timestamp["external"] = str(val).lower() in ("1", "true", "yes")

                points.append(PointStruct(
                    id=safe_id,
                    vector=vector,
                    payload=meta_with_timestamp
                ))

            if not points:
                return 0

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            logger.info(f"✅ 成功添加 {len(points)} 个向量")
            return len(points)

        except Exception as e:
            logger.error(f"❌ 添加向量失败: {e}")
            return 0

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似向量 - 使用新版 query_points() API
        """
        try:
            if len(query_vector) != self.vector_size:
                logger.error(f"❌ 查询向量维度错误: 期望{self.vector_size}, 实际{len(query_vector)}")
                return []

            # 构建过滤器
            query_filter = None
            if where:
                conditions = []
                for key, value in where.items():
                    if isinstance(value, (str, int, float, bool)):
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value)
                            )
                        )
                if conditions:
                    query_filter = Filter(must=conditions)

            # 搜索参数
            search_params = None
            try:
                search_params = models.SearchParams(
                    hnsw_ef=self.search_ef,
                    exact=self.search_exact
                )
            except Exception:
                search_params = None

            # 使用 query_points()，直接传入向量列表
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,                     # 直接传入向量
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
                search_params=search_params
            )

            # 转换结果
            results = []
            for hit in response.points:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "metadata": hit.payload or {}
                })

            logger.debug(f"🔍 Qdrant搜索返回 {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"❌ 向量搜索失败: {e}")
            return []

    def delete_vectors(self, ids: List[Union[int, str]]) -> bool:
        try:
            if not ids:
                return True
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=ids),
                wait=True
            )
            logger.info(f"✅ 删除 {len(ids)} 个向量")
            return True
        except Exception as e:
            logger.error(f"❌ 删除失败: {e}")
            return False

    def clear_collection(self) -> bool:
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
            logger.info(f"✅ 清空集合: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"❌ 清空失败: {e}")
            return False

    def delete_memories(self, memory_ids: List[str]):
        try:
            if not memory_ids:
                return
            conditions = [
                FieldCondition(key="memory_id", match=MatchValue(value=mid))
                for mid in memory_ids
            ]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=Filter(should=conditions)),
                wait=True,
            )
            logger.info(f"✅ 按memory_id删除 {len(memory_ids)} 个向量")
        except Exception as e:
            logger.error(f"❌ 删除记忆失败: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection_name)
            vectors_count = getattr(info, "vectors_count", getattr(info, "points_count", 0))
            indexed = getattr(info, "indexed_vectors_count", 0)
            points = getattr(info, "points_count", 0)
            segments = getattr(info, "segments_count", 0)

            return {
                "name": self.collection_name,
                "vectors_count": vectors_count,
                "indexed_vectors_count": indexed,
                "points_count": points,
                "segments_count": segments,
                "config": {
                    "vector_size": self.vector_size,
                    "distance": self.distance.value,
                }
            }
        except Exception as e:
            logger.error(f"❌ 获取集合信息失败: {e}")
            return {}

    def get_collection_stats(self) -> Dict[str, Any]:
        info = self.get_collection_info()
        info["store_type"] = "qdrant"
        return info

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"❌ 健康检查失败: {e}")
            return False

    def __del__(self):
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
            except Exception:
                pass


# ========== 测试 ==========
if __name__ == "__main__":
    import os
    import numpy as np
    import logging
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    store = QdrantVectorStore(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=os.getenv("QDRANT_COLLECTION", "hello_agents_vectors"),
        vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", 384)),
        distance=os.getenv("QDRANT_DISTANCE", "cosine"),
        timeout=int(os.getenv("QDRANT_TIMEOUT", 30))
    )

    # 生成示例数据
    vectors = [np.random.rand(store.vector_size).tolist() for _ in range(5)]
    metadata = [{"name": f"item_{i}", "category": "test"} for i in range(5)]
    custom_ids = [f"custom_id_{i}" for i in range(5)]   # 非法ID，将被替换

    added = store.add_vectors(vectors, metadata, ids=custom_ids)
    print(f"成功添加向量数量: {added}")

    # 搜索
    query = np.random.rand(store.vector_size).tolist()
    results = store.search_similar(query, limit=3)
    for res in results:
        print(f"ID: {res['id']}, Score: {res['score']:.4f}, Meta: {res['metadata']}")

    info = store.get_collection_info()
    print(f"集合信息: {info}")