import asyncio
from hello_agents.protocols import MCPClient
async def connect_to_server():
    #1.
    #npx
    client = MCPClient([
        "npx", "-y", "@modelcontextprotocol/server-github",
        "." #指定目录
    ])
    #使用async with 确保正确连接
    async with client:
        #使用client
        tools = await client.list_tools()
        print(f"可用工具{[t['name'] for t in tools]}")

    #2·
    client2 = MCPClient(['python', 'my_server.py'])
    async with client2:
        pass

asyncio.run(connect_to_server())

