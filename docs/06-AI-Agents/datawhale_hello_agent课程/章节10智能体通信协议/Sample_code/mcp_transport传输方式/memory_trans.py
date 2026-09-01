from hello_agents.tools import MCPTool
# 1. Memory Transport - 内存传输（用于测试）

mcp_tool = MCPTool()

# 列出可用工具
result = mcp_tool.run({"action": "list_tools"})
print(f"可用工具：{result}")

# use tool
result = mcp_tool.run({
    "action": "call_tool",
    "tool_name": "add",
    "arguments": {"a": 5, "b": 3}
})
print(result)