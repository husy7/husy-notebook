from typing import List, Dict, Any, Optional

class Memory:

    def __init__(self):

        """
        初始化方法
        初始化一个空的内存列表，用于存储记忆数据
        每个记忆数据是一个字典，包含字符串键和任意值
        """
        self.records: List[Dict[str, Any]] = []  # 初始化一个空列表，用于存储记忆数据

    
    def add_record(self, record_tyoe:str, content:str):
        """
        添加记忆记录
        :param record_tyoe: 记忆类型，例如“事件”、“情感”等
        :param content: 记忆内容，例如代码生成反馈等
        """
        record = {"type": record_tyoe, "content": content}
        self.records.append(record)
        print(f"记忆更新，新增{record_tyoe}类型记录：{content}")


    def get_trajectory(self):
        """
        获取记忆轨迹
        :return: 返回一个列表，包含所有记忆记录
        """
        trajectory_parts = []
        for record in self.records:
            if record['type'] == 'execution':
                trajectory_parts.append(f"--- 上一轮尝试 (代码) ---\n{record['content']}")
            elif record['type'] == 'reflection':
                trajectory_parts.append(f"--- 评审员反馈 ---\n{record['content']}")
        
        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:
        """
        获取最后一次反思内容
        :return: 返回最后一次反思的内容，如果没有反思记录则返回None
        """
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None


