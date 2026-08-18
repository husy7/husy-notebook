import os
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM
from typing import List, Dict, Any, Optional

load_dotenv()

DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}


class Memory:
    """记忆管理类，存储执行和反思的历史记录"""
    
    def __init__(self):
        """初始化一个空的内存列表，用于存储记忆数据"""
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        添加记忆记录
        :param record_type: 记忆类型，"execution" 或 "reflection"
        :param content: 记忆内容
        """
        record = {"type": record_type, "content": content}
        self.records.append(record)
        # 截断显示，避免输出过长
        display_content = content[:100] + "..." if len(content) > 100 else content
        print(f"记忆更新，新增{record_type}类型记录：{display_content}")

    def get_trajectory(self) -> str:
        """获取完整的记忆轨迹"""
        trajectory_parts = []
        for record in self.records:
            if record['type'] == 'execution':
                trajectory_parts.append(f"--- 执行记录 ---\n{record['content']}")
            elif record['type'] == 'reflection':
                trajectory_parts.append(f"--- 反思反馈 ---\n{record['content']}")
        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:
        """获取最后一次执行记录的内容"""
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None

    def get_last_reflection(self) -> Optional[str]:
        """获取最后一次反思内容"""
        for record in reversed(self.records):
            if record['type'] == 'reflection':
                return record['content']
        return None
    
    def get_execution_count(self) -> int:
        """获取执行记录的数量"""
        return sum(1 for record in self.records if record['type'] == 'execution')
    
    def clear(self):
        """清空所有记忆"""
        self.records.clear()
        print("记忆已清空")


class ReflectionAgent:
    """
    反思代理：通过迭代执行-反思-优化循环来改进回答
    
    工作流程：
    1. 初始执行：根据任务生成初始回答
    2. 反思阶段：分析当前回答的质量
    3. 优化阶段：根据反馈改进回答
    4. 重复2-3步直到达到最大迭代次数或无需改进
    """
    
    def __init__(self, llm_client, max_iterations: int = 3, verbose: bool = True):
        """
        初始化反思代理
        
        :param llm_client: HelloAgentsLLM 实例
        :param max_iterations: 最大迭代次数
        :param verbose: 是否打印详细日志
        """
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations
        self.verbose = verbose

    def run(self, task: str) -> str:
        """
        执行反思-优化循环
        
        :param task: 任务描述
        :return: 最终的回答
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"开始处理任务")
            print(f"{'='*60}")
            print(f"任务: {task}\n")

        # --- 1. 初始执行 ---
        if self.verbose:
            print("--- 第 1 阶段: 初始执行 ---")
        
        initial_prompt = DEFAULT_PROMPTS["initial"].format(task=task)
        initial_response = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_response)

        # --- 2. 迭代循环: 反思与优化 ---
        for i in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"第 {i+1}/{self.max_iterations} 轮迭代")
                print(f"{'='*60}")

            # a. 反思阶段
            if self.verbose:
                print("\n→ 反思阶段: 分析当前回答...")
            
            last_response = self.memory.get_last_execution()
            if last_response is None:
                print("⚠️ 没有找到之前的执行记录，跳过反思")
                break
                
            reflect_prompt = DEFAULT_PROMPTS["reflect"].format(
                task=task, 
                content=last_response
            )
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)

            # b. 检查是否需要停止
            if "无需改进" in feedback:
                if self.verbose:
                    print("\n✅ 回答已无需改进，任务完成！")
                break

            # c. 优化阶段
            if self.verbose:
                print("\n→ 优化阶段: 根据反馈改进回答...")
            
            refine_prompt = DEFAULT_PROMPTS["refine"].format(
                task=task,
                last_attempt=last_response,
                feedback=feedback
            )
            refined_response = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_response)
            
            if self.verbose:
                print(f"✅ 第 {i+1} 轮迭代完成")

        # --- 3. 返回最终结果 ---
        final_response = self.memory.get_last_execution()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("任务完成！")
            print(f"{'='*60}")
            print(f"总执行次数: {self.memory.get_execution_count()}")
            print(f"总迭代轮次: {i+1}")
            print(f"\n最终回答:\n{'-'*40}\n{final_response}")
        
        return final_response

    def _get_llm_response(self, prompt: str) -> str:
        """
        调用LLM并获取完整的响应
        
        :param prompt: 提示词
        :return: LLM的响应文本
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.think(messages=messages)
            
            # 处理不同类型的响应
            if response is None:
                return ""
            
            # 如果返回的是字符串，直接返回
            if isinstance(response, str):
                return response
            
            # 如果返回的是生成器或迭代器，收集所有片段
            if hasattr(response, '__iter__') and not isinstance(response, (bytes, dict)):
                try:
                    full_response = ""
                    for chunk in response:
                        if chunk:
                            if isinstance(chunk, str):
                                full_response += chunk
                            else:
                                full_response += str(chunk)
                    return full_response
                except Exception as e:
                    if self.verbose:
                        print(f"警告：处理LLM响应时出错: {e}")
                    return str(response)
            
            # 其他情况，尝试转换为字符串
            return str(response)
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 调用LLM时出错: {e}")
            return f"错误：{str(e)}"

    def get_trajectory(self) -> str:
        """获取完整的执行轨迹"""
        return self.memory.get_trajectory()
    
    def reset(self):
        """重置代理状态（清空记忆）"""
        self.memory.clear()
        if self.verbose:
            print("代理已重置")


# 使用示例
if __name__ == "__main__":
    # 初始化LLM
    print("初始化 HelloAgentsLLM...")
    llm = HelloAgentsLLM()
    
    # 创建反思代理
    agent = ReflectionAgent(
        llm_client=llm, 
        max_iterations=3,
        verbose=True  # 设置为False可减少输出
    )
    
    # 运行任务
    task = "任务：编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
    result = agent.run(task)
    
    # 可选：查看完整的执行轨迹
    # print("\n完整执行轨迹:")
    # print(agent.get_trajectory())