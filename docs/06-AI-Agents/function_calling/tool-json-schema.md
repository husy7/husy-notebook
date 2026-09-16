```python
{
  "type": "function",           ← 固定值，永远是它
  "function": {
    "name": "<标识符>",          ← 你代码里分发用的 key
    "description": "<何时用>",    ← 给模型的决策依据
    "parameters": {
      "type": "object",          ← 固定值，永远是它
      "properties": {
        "<参数名>": {"type": "<类型>"}
      },
      "required": ["<参数名>"]
    }
  }
}

```
