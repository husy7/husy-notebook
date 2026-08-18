# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""

import ast
import re
from hello_agents import HelloAgentsLLM

class PlanSolveAgent:
    def __init__(self, llm_client: HelloAgentsLLM, question: str):
        self.llm_client = llm_client
        self.question = question

    def _generate_plan(self) -> list:
        prompt = DEFAULT_PLANNER_PROMPT.format(question=self.question)
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.invoke(messages) or ""
        # 尝试多种解析方式
        try:
            # 先尝试直接解析 (如果模型仅输出列表)
            plan = ast.literal_eval(response.strip())
            if isinstance(plan, list):
                return plan
        except:
            pass
        # 尝试从代码块中提取
        match = re.search(r"```python\s*([\s\S]*?)\s*```", response)
        if match:
            try:
                plan = ast.literal_eval(match.group(1).strip())
                if isinstance(plan, list):
                    return plan
            except:
                pass
        # 尝试正则提取方括号内容
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            try:
                plan = ast.literal_eval(match.group())
                if isinstance(plan, list):
                    return plan
            except:
                pass
        
        return []

    def run(self) -> str:
        plan = self._generate_plan()
        if not plan:
            return "无法生成有效计划，请检查输入或重试。"
        
        history = ""
        final_answer = ""
        for i, step in enumerate(plan):
            prompt = DEFAULT_EXECUTOR_PROMPT.format(
                question=self.question,
                plan=plan,
                history=history if history else "无",
                current_step=step
            )
            messages = [{"role": "user", "content": prompt}]
            result = self.llm_client.invoke(messages) or ""
            # 只保留最近一步的结果作为上下文
            history = f"上一步结果: {result}"
            final_answer = result
            print(f"步骤 {i+1}/{len(plan)} 完成: {step}\n结果: {result}")
        return final_answer

if __name__ == "__main__":
    llm = HelloAgentsLLM()
    question = "计算：2+3*10"
    agent = PlanSolveAgent(llm, question)
    answer = agent.run()
    print("最终答案:", answer)