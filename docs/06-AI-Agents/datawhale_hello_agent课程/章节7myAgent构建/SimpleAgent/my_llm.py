import os
from typing import Optional
from openai import OpenAI
from hello_agents import HelloAgentsLLM

class MyLLM(HelloAgentsLLM):
    '''
    一个自定义的LLM客户端，通过继承增加了对ModelScope的支持。
    
    '''
    def __init__(
        self,
        model:Optional[str] =None,
        api_key:Optional[str] = None,
        base_url:Optional[str] = None,
        provider:Optional[str] = "auto",#
        **kwargs):
        if provider == 'modelscope':
            # 使用ModelScope的API
            self.provider = provider
            self.api_key = api_key or os.getenv('MODELSCOPE_API_KEY')
            self.base_url = base_url or "https://api-inference.modelscope.cn/v1/"

            if not self.api_key:
                raise ValueError('API key is required for ModelScope')

            #
            self.model = model or os.getenv('LLM_MODEL_ID')
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)


            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            super().__init__(model = model, api_key=api_key, base_url=base_url, provider = provider, **kwargs)

        