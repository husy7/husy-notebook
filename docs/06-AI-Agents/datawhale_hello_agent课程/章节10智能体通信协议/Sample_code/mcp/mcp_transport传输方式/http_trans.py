'''
HTTP Transport - HTTP 传输

适用场景：生产环境、远程服务、微服务架构

'''

# 注意：MCPTool 主要用于 Stdio 和 Memory 传输
# 对于 HTTP/SSE 等远程传输，建议使用底层的 MCPClient
import asyncio
from hello_agents.protocols import MCPClient
async def http_transport():
    # 连接到远程HTTP MCP 服务器
    client = MCPClient("http:api.example.com/mcp")

    async with client:
        # 获取信息
        tools = await client.list_tools()
        print(f"服务器提供了 {len(tools)} 个工具：")
        # 使用工具
        result = await client.call_tool('process_data',{
            "data":"Hello World",
            "operation":"reverse"
        })
        print(f"处理结果：{result}")
# 需要实际http服务器支持MCP协议才能运行
#asyncio.run(http_transport())