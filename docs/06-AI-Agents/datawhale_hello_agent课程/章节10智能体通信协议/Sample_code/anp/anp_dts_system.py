'''构建一个完整的分布式任务调度系统'''


from hello_agents.protocols import ANPDiscovery, register_service
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools.builtin import ANPTool
import random
from dotenv import load_dotenv

load_dotenv()
llm = HelloAgentsLLM()

# 1. 创建服务发现中心
discovery = ANPDiscovery()


# 2. 注册任务调度节点
for i in range(10):
    register_service(
        discovery=discovery,
        service_id=f"comput_node_{i}",
        service_name=f"计算节点{i}",
        service_type="compute_node",
        capabilities=["data_processing", "ml_training"],
        endpoint=f"http://node{i}:8000",
        metadata={
            "load": random.uniform(0.1, 0.9),
            "cpu_cores": random.choice([4, 8, 16]),
            "memory_gb": random.choice([16, 32, 64]),
            "gpu": random.choice([True, False])
        }
    )
print(f"✅ 注册了 {len(discovery.list_all_services())} 个计算节点")

# 3. 创建任务调度Agent
scheduler = SimpleAgent(
    name="任务调度Agent",
    llm=llm,
    system_prompt="""你是一个智能任务调度器，负责：
1. 分析任务需求
2. 选择最合适的计算节点
3. 分配任务

选择节点时考虑：负载、CPU核心数、内存、GPU等因素。"""
)

# 添加anp工具
anp_tool = ANPTool(
    name="service_discovery",
    description="用于发现和调用分布式计算节点服务",
    discovery=discovery
)
scheduler.add_tool(anp_tool)

# 4.任务分配
def assign_task(task_description: str):
    print(f"接收到任务: {task_description}")
    print('=' * 50)

    # 让agent选择节点
    response = scheduler.run(
        f"""请为以下任务选择最合适的计算节点:
        {task_description}
        要求：
        1. 列出所有可用节点
        2. 分析每个节点的特点
        3. 选择最合适的节点
        4. 说明选择理由
"""
    )

    print(response)
    print('=' * 50)

# 示例任务
assign_task("训练一个深度学习模型，数据量大，需要GPU支持。")
assign_task("进行大规模数据处理，要求高内存和多核CPU。")
assign_task("运行一个轻量级的机器学习任务，对资源要求不高。")