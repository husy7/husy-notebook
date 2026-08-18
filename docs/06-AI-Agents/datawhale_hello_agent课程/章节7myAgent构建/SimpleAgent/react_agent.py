import re
from typing import Optional, List, Tuple
from hello_agents import ReActAgent, HelloAgentsLLM, Config, Message, ToolRegistry
from  react_prompt import MY_REACT_PROMPT
class MyReactAgent(ReActAgent):
    '''
    一个自定义的Agent，通过继承增加了对ModelScope的支持。
    
    '''
    def __int__(
            self,
            name:str,
            llm:HelloAgentsLLM,
            tool_regiter:ToolRegistry,
            system_prompt:str = None,
            config:Optional[Config] = None,
            max_steps:int = 5,
            custom_prompt:Optional[str] = None,
            **kwargs):
        super().__init__(name, llm, system_prompt, config)
        self.tool_regiter = tool_regiter
        self.max_steps = max_steps
        self.current_history = List[str] = None
        self.prompt_template = custom_prompt if custom_prompt else MY_REACT_PROMPT
        print(f'{name}初始化完成, max_steps={max_steps}\n prompt_template={self.prompt_template}')

    def run(self, input_text: str, **kwargs) -> str:
        """运行ReAct Agent"""
        self.current_history = []
        current_step = 0

        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # 1. 构建提示词
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str
            )

            # 2. 调用LLM
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.invoke(messages, **kwargs)

            # 3. 解析输出
            thought, action = self._parse_output(response_text)

            # 4. 检查完成条件
            if action and action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                return final_answer

            # 5. 执行工具调用
            if action:
                tool_name, tool_input = self._parse_action(action)
                observation = self.tool_registry.execute_tool(tool_name, tool_input)
                self.current_history.append(f"Action: {action}")
                self.current_history.append(f"Observation: {observation}")

        # 达到最大步数
        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer
