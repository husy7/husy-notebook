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

#asyncio.run(connect_to_server())

async def discover_tools():
    client = MCPClient(["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])

    async with client:
        # 获取所有可用工具
        tools = await client.list_tools()

        print(f"服务器提供了 {len(tools)} 个工具：")
        for tool in tools:
            print(f"\n工具名称: {tool['name']}")
            print(f"描述: {tool.get('description', '无描述')}")

            # 打印参数信息
            if 'inputSchema' in tool:
                schema = tool['inputSchema']
                if 'properties' in schema:
                    print("参数:")
                    for param_name, param_info in schema['properties'].items():
                        param_type = param_info.get('type', 'any')
                        param_desc = param_info.get('description', '')
                        print(f"  - {param_name} ({param_type}): {param_desc}")

asyncio.run(discover_tools())

# 输出示例：
# 服务器提供了 5 个工具：
#
# 工具名称: read_file
# 描述: 读取文件内容
# 参数:
#   - path (string): 文件路径
#
# 工具名称: write_file
# 描述: 写入文件内容
# 参数:
#   - path (string): 文件路径
#   - content (string): 文件内容
