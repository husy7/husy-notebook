import os
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from dotenv import load_dotenv
load_dotenv()
def test_neo4j_connection():
    # 从环境变量读取连接信息（推荐）
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    # 如果未设置环境变量，可在此直接指定（仅供测试）
    # uri = "neo4j+s://xxxxx.databases.neo4j.io"
    # username = "neo4j"
    # password = "your_password"

    if not uri or not username or not password:
        print("❌ 错误：请设置 NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD 环境变量")
        return False

    driver = None
    try:
        # 创建驱动实例[reference:14][reference:15]
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # 验证连接是否成功建立[reference:16][reference:17][reference:18]
        driver.verify_connectivity()
        print("✅ 连接成功！")
        
        # 可选：执行一个简单查询验证数据库可访问性
        with driver.session() as session:
            result = session.run("RETURN 1 AS num")
            record = result.single()
            print(f"数据库响应测试查询结果：{record['num']}")
        
        return True

    except AuthError as e:
        print(f"❌ 认证失败，请检查用户名或密码：{e}")
        return False
    except ServiceUnavailable as e:
        print(f"❌ 服务不可用，请检查 URI 或网络连接：{e}")
        return False
    except Exception as e:
        print(f"❌ 发生未知异常：{e}")
        return False
    finally:
        # 确保驱动被正确关闭[reference:19][reference:20]
        if driver:
            driver.close()

if __name__ == "__main__":
    test_neo4j_connection()