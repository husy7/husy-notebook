import os
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from dotenv import load_dotenv
load_dotenv()
def test_qdrant_connection():
    # 从环境变量读取（推荐）
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")  # 无 API 密钥时可留空

    # 如果未设置环境变量，可在此直接指定（仅供测试）
    # url = "https://your-cluster.qdrant.tech:6333"
    # api_key = "your_api_key_here"

    if not url:
        print("❌ 错误：未设置 QDRANT_URL 环境变量")
        return False

    try:
        # 初始化客户端
        client = QdrantClient(url=url, api_key=api_key)

        # 执行简单请求：获取所有集合列表（可证明连通性）
        collections = client.get_collections()
        print("✅ 连接成功！")
        print(f"当前集合列表：{collections}")
        return True

    except UnexpectedResponse as e:
        print(f"❌ 连接失败（服务端返回错误）：{e}")
        return False
    except Exception as e:
        print(f"❌ 发生异常：{e}")
        return False

if __name__ == "__main__":
    test_qdrant_connection()