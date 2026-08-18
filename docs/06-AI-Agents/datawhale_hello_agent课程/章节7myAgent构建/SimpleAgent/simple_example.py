from hello_agents import SimpleAgent, HelloAgentsLLM
from dotenv import load_dotenv
load_dotenv()

llm = HelloAgentsLLM()

agent = SimpleAgent(
    name="助手",
    llm=llm,
    system_prompt="你加大赢鲸鱼，是一个细心的ai助手"

)

# base communite
response = agent.run("你好， 自我介绍一下")
print(response)


from hello_agents.tools import CalculatorTool
calculator = CalculatorTool()

response = agent.run("计算：2+3*10")

print(response)

print(f"history :{len(agent.get_history())}")

