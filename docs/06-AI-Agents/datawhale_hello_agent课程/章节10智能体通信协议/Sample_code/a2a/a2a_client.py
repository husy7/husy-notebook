from hello_agents.protocols import A2AClient

# 创建客户端连接到研究员Agent
client = A2AClient("http://localhost:8082")

# 发送研究请求
response = client.execute_skill("research", "研究人工智能在医疗领域的应用")
print("研究员智能体响应:", response.get('result'))

#输出
# 输出：
# 收到响应：{'topic': 'AI在医疗领域的应用', 'findings': '关于AI在医疗领域的应用的研究结果...', 'sources': ['来源1', '来源2', '来源3']}