from hello_agents.protocols import A2AServer
import threading
import time
# 创建研究员Agent服务
researcher = A2AServer(
    name="researcher",
    description="研究员智能体，负责提出问题和分析结果",
    version="1.0.0",
)

# 添加技能
@researcher.skill("research")
def handle_research(text: str) -> str:
    """处理研究请求"""
    import re
    match = re.search(r'research\s+(.+)', text, re.IGNORECASE)
    topic = match.group(1).strip() if match else text
    
    # 实际的研究逻辑（这里简化）
    result = {
        "topic": topic,
        "findings": f"关于{topic}的研究结果...",
        "sources": ["来源1", "来源2", "来源3"]
    }
    return str(result)

# 后台服务
def start_server():
    researcher.run(host="localhost", port=8082)

if __name__ == "__main__":
    # 启动研究员智能体服务
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    print("研究员智能体服务已启动，监听端口 8082")
    
    # 主线程可以继续执行其他任务
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("研究员智能体服务已停止")