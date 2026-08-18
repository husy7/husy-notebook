from typing import Optional, Iterator
import re
from hello_agents import SimpleAgent, HelloAgentsLLM, Config, ToolRegistry, Message

class MySimpleAgent(SimpleAgent):
    '''
    重写的简单对话Agent
    展示如何基于框架基类构建自定义Agent
    
    '''
    def __init__(
        self,
        name:str,
        llm:HelloAgentsLLM,
        system_prompt:Optional[str]=None,
        config:Optional[Config]=None,
        tool_registry:Optional['ToolRegistry']=None,
        enable_tool_calling:bool=True
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        state = '启用' if self.enable_tool_calling else '禁用'
        print(f"✅ {name} 初始化完成，工具调用: {state}")

    def run(self, input_text:str, max_tool_iterations:int = 3, **kwargs) -> str:
        '''
        重写run方法，实现对话，可以工具调用
        '''
        print(f"{self.name} 接收到输入: {input_text}")

        messages = []

        # 添加系统消息
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        #添加历史消息
        for msg in self._history:
            messages.append({'role':msg.role, 'content':input_text})

        messages.append({'role':'user', 'content':input_text})

        #没有使用工具，简单对话

        if not self.enable_tool_calling:
            respoonse = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, 'user'))
            self.add_message(Message(respoonse, 'assistant'))
            print(f'{self.name},相应完成')
            return respoonse

        # 如要使用工具
        return self._run_with_tools(messages,input_text, max_tool_iterations, **kwargs)

    def _get_enhanced_system_prompt(self):
        '''
        获取增强的系统提示,包含工具信息
        '''
        base_prompt = self.system_prompt or '你是帮助我的助手'

        if not self.tool_registry or not self.enable_tool_calling:
            return base_prompt

        tool_descriptions = self.tool_registry.get_tools_description()

        if not tool_descriptions or tool_descriptions == '暂无工具':

            return base_prompt

        tools_section = '\n\n##可以工具\n'
        tools_section += '你还有使用以下工具协助你完成回答：\n'
        tools_section += '\n'.join(tool_descriptions) + '\n'

        tools_section += '##工具使用方法\n'
        tools_section += '你可以使用以下命令调用工具：\n'
        tools_section += "'[TOOL_CALL:{tool_name}:{parameters}]'\n"
        tools_section += "例如：'[TOOL_CALL:search:Python编程]'或 '[TOOL_CALL:memory:recall=用户信息]'\n\n"

        tools_section +="##工工具调用结果自动插入对话，然后你可以基于结果继续回答\n"

        return base_prompt + tools_section

    def _run_with_tools(self, messages:list, input_text:str, max_tool_iterations:int, **kwargs) -> str:
        '''
        支持工具的的运行逻辑
        '''
        current_iteration = 0
        final_response = ''

        while current_iteration < max_tool_iterations:
            #use llm to generate response
            response = self.llm.invoke(messages, **kwargs)

            # 检查是否包含工具调用
            tool_call = self._parse_tool_calls(response)

            if tool_call:
                print(f'{self.name}检测到工具调用: {tool_call}')
                tool_results = []
                clean_response = response

                for call in tool_call:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    clean_response = clean_response.replace(call['original'], "")
                messages.append({"role":"user","content":clean_response} )

                tool_results_text = '\n\n'.join(tool_results)
                messages.append({"role":"assistant", "content":f"工具执行结果：\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"})

                current_iteration += 1
                continue

            final_response = response
            break
        #超过最大迭代次数，返回最终结果
        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(input_text, 'user'))
        self.add_message(Message(final_response, 'assistant'))
        print(f'{self.name} 相应完成')
        return final_response

    def _parse_tool_calls(self, text:str) -> list:
        '''
        解析工具调用
        '''
        pattern = r"\[TOOL_CALL:(.*?)\:(.*?)\]"
        matches = re.findall(pattern, text)

        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append({
                'tool_name': tool_name,
                'parameters': parameters.strip(),
                'original': f"[TOOL_CALL:{tool_name}:{parameters}]"
            })

        return tool_calls


    def _execute_tool_call(self, tool_name:str, parameters:str) -> str:
        '''
        执行工具调用
        '''

        if not self.tool_registry:
            raise ValueError("没有工具注册")

        try:
            if tool_name == 'calculator':
                result = self.tool_registry.execute_tool(tool_name, parameters)
            else:
                param_dict = self._parse_tool_paramters(tool_name, parameters)
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    raise ValueError(f"没有找到工具: {tool_name}")
                result = tool.run(param_dict)
                return f"工具{tool_name}执行结果:\n {result}"
        except Exception as e:
            return f"执行工具时出错: {str(e)}"
       

    def _parse_tool_paramters(self, tool_name:str, parameters:str) -> dict:
        '''
        解析工具参数
        '''
        param_dict = {}

        if '=' in parameters:
            # 格式: key=value 或 action=search,query=Python
            if ',' in parameters:
                # 多个参数:action=search,query=Python,limit=3
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        param_dict[key.strip()] = value.strip()
            else:
                # 单个参数:key=value
                key, value = parameters.split('=', 1)
                param_dict[key.strip()] = value.strip()
        else:
            # 直接传入参数，根据工具类型智能推断
            if tool_name == 'search':
                param_dict = {'query': parameters}
            elif tool_name == 'memory':
                param_dict = {'action': 'search', 'query': parameters}
            else:
                param_dict = {'input': parameters}

        return param_dict

    def stream_run(self, input_text, **kwargs):
        '''
        自定义流式运行逻辑
        
        '''
        print(f'{self.name} 开始流式响应{input_text}')
        messages = []


        if self.system_prompt:
            messages.append({"role":"system","content":self.system_prompt})
        for meg in self._history:
            messages.append({'role':meg.role,'content':meg.content})

        messages.append({"role":"user","content":input_text})


        #streaming llm response
        full_response = ""
        print('实时响应')
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            
            full_response += chunk
            print(chunk, end="",flush=True)
            yield chunk

        print('\n')
        self.add_message(Message(input_text, 'user'))
        self.add_message(Message(full_response, 'assistant'))
        print(f'{self.name} 流式响应完成')
        #return full_response

    def add_tool(self, tool) -> None:
        '''
        添加工具
        '''
        if not self.tool_registry:
            from hello_agents import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        self.tool_registry.register_tool(tool)
        print(f"🔧 工具 '{tool.name}' 已添加")

    def has_tool(self)->bool:
        '''
        是否有工具
        '''
        return self.enable_tool_calling and self.tool_registry is not None

    
    def remove_tool(self, tool_name: str) -> bool:
        """移除工具（便利方法）"""
        if self.tool_registry:
            self.tool_registry.unregister(tool_name)
            return True
        return False    

    def list_tools(self) -> list:
        """列出所有可用工具"""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []