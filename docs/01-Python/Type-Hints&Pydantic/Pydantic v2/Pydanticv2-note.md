---
title: "Pydantic v2 数据校验（BaseModel）"
tags: [Python, Pydantic, 数据校验, 类型注解]
date: 2026-08-27
---
- (笔记内容依据：Pydantic v2 官方文档 docs.pydantic.dev，核心 API 为 v2 正式版语法)
# Pydantic v2 数据校验（BaseModel）
> 用 Python 类型注解声明数据结构，Pydantic 在运行时自动完成**解析、校验、类型转换**，
> 让看着像数据、实际是陷阱的输入——类型不对、格式不对、业务上离谱、该有的没有、不该有的乱塞的数据在进入系统入口时就被拦截—，不用再手写大量if/else 检查。
## 原理 / 动机
- **解决什么实际问题**：
  当外部输入（JSON、API 请求、配置、数据库行）是不可信时候。传统做法是手写校验代码，
  但复杂又容易漏。Pydantic 让你只写一次类型注解，校验逻辑自动生成。
- **核心原理（简洁全面）**：
  1. `BaseModel` 子类在**类创建时**读取类型注解，用 Rust 写的校验核心（pydantic-core）
     为每个字段生成一个 Schema；
  2. 实例化时数据流经该 Schema：先做**宽松模式解析**（如 `"123"` → `123`），再做约束校验
     （`Field(ge=0, le=150)`、正则等）；
  3. 全部通过 → 得到类型**确定**的对象；任一失败 → 抛出结构化的 `ValidationError`，
     包含所有错误（不是只报第一个）。
- **为什么必须这样设计**：
  校验核心用 Rust 重写（v1 是纯 Python），性能提升 5~50 倍；"声明式注解"让模型本身
  就是一份可读的接口文档，还能自动生成 JSON Schema。失败时收集**全部**错误而非快速失败，
  是为了方便前端一次性提示用户所有问题。
## 怎么应用示例
- **适用场景**：FastAPI 请求/响应模型、配置管理（`pydantic-settings`）、爬虫/ETL 数据清洗、
  任何"外部数据进入系统"的边界。
- **快速上手步骤**：
  1. 定义继承 `BaseModel` 的类，给字段写类型注解和约束；
  2. 用字典实例化（或 `model_validate(json_str)`），自动校验+转换；
  3. 用 `model_dump()` 序列化；失败时捕获 `ValidationError` 处理。
```python
# pip install "pydantic>=2"
from datetime import datetime
from pydantic import (BaseModel, Field, ConfigDict,
                      field_validator, model_validator,
                      computed_field, ValidationError)
class User(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    id: int
    name: str = Field(min_length=1, max_length=20)
    email: str
    age: int = Field(ge=0, le=150)
    tags: list[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    # 单字段校验器（v2 新写法，替代 v1 的 @validator）
    @field_validator('email')
    @classmethod
    def check_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('must contain @')
        return v
    # 多字段联合校验（v2 新写法，替代 v1 的 @root_validator）
    @model_validator(mode='after')
    def check_consistency(self):
        if self.age < 18 and 'adult' in self.tags:
            raise ValueError('未成年人不能带 adult 标签')
        return self
    # 计算字段：校验后派生出的值，会出现在 model_dump() 里
    @computed_field
    @property
    def is_adult(self) -> bool:
        return self.age >= 18
# 1) 宽松解析：字符串 "123" 自动转 int，name 两侧空白被去掉
u = User(id="123", name=" alice ", email="a@b.com", age=30)
print(u.id, type(u.id))   # 123 <class 'int'>
print(u.is_adult)         # True
# 2) 从 dict / JSON 反序列化（v2 新 API：model_validate）
u2 = User.model_validate({"id": 1, "name": "bob",
                          "email": "b@c.com", "age": "20"})
print(u2.model_dump())    # 序列化为 dict（v2 新 API，替代 v1 的 .dict()）
# 3) 校验失败：一次性收集全部错误，而不是只报第一个
try:
    User(id=1, name="x", email="bad-email", age=200)
except ValidationError as e:
    for err in e.errors():
        print(err['type'], err['loc'])  # value_error ('email',) / less_than_equal ('age',)
```
## 边界 / 常见坑
- ❌ **错误现象**：还在用 v1 的 API（`@validator`、`@root_validator`、`.dict()`、`.json()`、
  `parse_obj()`），升级 v2 后报 `DeprecationWarning` 或直接 `AttributeError`。
  ✅ **正确做法**：换成 v2 API —— `@field_validator`、`@model_validator(mode='after')`、
  `model_dump()`、`model_dump_json()`、`model_validate()`。
- ❌ **错误现象**：以为 `age: int = 1` 是"必填且默认 1"——实际是**可省略**（有默认值就不必填）；
  想要可空字段写 `Optional[int]` 却发现 `None` 会传进去。
  ✅ **正确做法**：必填就不给默认值；`Optional[int] = None` 表示"可为 None 且默认 None"；
  若要求"必须显式传入（哪怕是 None）"，用 `Field(...)` 或 v2.7+ 的
  `model_config = ConfigDict(validate_default=True)` 配合校验器。
- ❌ **错误现象**：把 `float` 类型的 `1.5` 传给 `int` 字段，v1 会悄悄截断成 `1`，
  v2 默认会**报错**（v2 更严格）。
  ✅ **正确做法**：数据源确实有小数时用 `float`，或在字段上用自定义转换器处理。
- **边界条件**：
  - 校验只在**实例化/反序列化时**发生，之后直接改属性（`u.age = -1`）默认**不触发校验**，
    需要 `model_config = ConfigDict(validate_assignment=True)`；
  - 默认是宽松模式（str→int 会转），要 Python 原生严格行为用 `strict=True`；
  - 复杂业务规则（如"查数据库判断唯一性"）不适合放校验器里，那是业务层的职责。
## 关联 / 类比
- 前置知识：[[Python 类型注解]]、[[dataclass]]
- 类似概念：[[dataclass]]（区别是 dataclass 只做存储不校验、不转换类型，是"哑"容器）；
  [[marshmallow]]（区别是 marshmallow 需要为序列化和反序列化各写一套 Schema，Pydantic 一份注解通吃）
- 生活类比：像机场安检——所有行李（数据）必须过同一台扫描仪（Schema），
  违禁品（非法值）当场没收并给你一张清单（ValidationError 列出所有问题），
  通过后行李被贴上标准标签（类型确定的对象）才能登机（进入系统）。
## 自我检验
1. **一句话说清**：它用类型注解在运行时自动校验和转换外部输入，把"防御脏数据"的代码从手写变成声明式。核心动机：输入不可信、手写校验啰嗦易漏。
2. **最小示例**：`class M(BaseModel): x: int` → `M(x="3").x == 3`，两行就够了。
3. **常见错误**：混用 v1/v2 API（`.dict()` vs `model_dump()`）；以为有默认值的字段还是必填；忘记 `validate_assignment=True` 导致事后修改不校验。
4. **与已会概念的区别**：和 `dataclass` 像（都是注解驱动的数据类），区别是 dataclass 无校验无转换，Pydantic 是"带安检门的 dataclass"。
## 对比与选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| Pydantic v2（本文） | 类型注解即 Schema，Rust 核心高性能校验+转换 | API 边界、配置管理、需要 JSON Schema 的场景 |
| dataclasses（标准库） | 零依赖的注解数据容器，无校验 | 内部纯数据传递、不想引入第三方依赖 |
| attrs | 更灵活的类构建器（slots、validators 可选） | 需要精细控制类行为（slots、不可变对象） |
| marshmallow | 显式定义序列化 Schema，校验/序列化分离 | 老项目、需要"加载/转储"两套不同 Schema |
## 参考
- [Pydantic v2 官方文档 - Validation](https://docs.pydantic.dev/latest/concepts/validation/)
- [Pydantic v2 官方文档 - Validators](https://docs.pydantic.dev/latest/concepts/validators/)
- [Pydantic v2 官方文档 - Migration Guide (v1→v2)](https://docs.pydantic.dev/latest/migration/)
- [应用示例pydanticv2_sample.py](pydanticv2_sample.py)