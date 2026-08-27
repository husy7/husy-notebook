"""工具基本类"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel


class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None

    @property
    def schema(self) -> Dict[str, Any]:
        """返回参数的最小结构，供需要 JSON Schema 的调用方使用。"""
        return {"name": self.name, "type": self.type, "required": self.required}


class Tool(ABC):
    """工具基类

    模板方法：run() 负责按需校验参数并委托给 _execute() 执行。
    子类需要实现 get_parameters()、_execute() 以及（可选）validate_parameters()。
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """校验参数

        默认实现：检查所有必需的参数字段是否齐全。
        子类可覆写以增加更严格的动作级校验。
        """
        required_params = [p.name for p in self.get_parameters() if p.required]
        return all(param in parameters for param in required_params)

    def run(self, parameters: Dict[str, Any]) -> Any:
        """执行工具的核心逻辑

        模板方法：先校验，再委托给子类实现的 _execute()。

        Args:
            parameters: 运行参数

        Returns:
            Any: 执行结果
        """
        if not self.validate_parameters(parameters):
            return "❌ 参数验证失败：缺少必需的参数"
        return self._execute(parameters)

    @abstractmethod
    def _execute(self, parameters: Dict[str, Any]) -> Any:
        """子类需要实现的实际执行逻辑"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.dict() for p in self.get_parameters()]
        }

    def to_str(self) -> str:
        return f"Tool(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()
