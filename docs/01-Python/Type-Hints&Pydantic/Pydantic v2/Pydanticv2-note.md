---
title: "Pydantic v2 数据校验（BaseModel）"
tags: [Python, Pydantic, 数据校验, 类型注解]
date: 2026-08-27
---

# Pydantic v2 数据校验（BaseModel）

## 定义
> 笔记内容依据：Pydantic v2 官方文档（docs.pydantic.dev），核心 API 均为 v2 正式版语法。

Pydantic v2 是一个基于 Python 类型注解的**运行时数据校验（validation）与解析（parsing）**库：你只需继承 `BaseModel` 声明一个数据类，给每个字段标注类型并挂上约束（`Field(ge=0, le=150)`、字段/模型校验器等），库就会在对象创建时自动完成数据清洗，让"看着像数据、实际是陷阱"的输入在进入系统入口时就被拦截。

它解决的核心问题是：当外部输入（JSON、API 请求、配置、数据库行）**不可信**时，传统做法是手写大量 `if/else` 校验代码——复杂、啰嗦、容易漏。Pydantic 让你只写一次类型注解，校验逻辑自动生成，无需再手写防御性检查。

核心特征包括：① 声明式——类型注解即 Schema，模型本身就是一份可读的接口文档，还能自动生成 JSON Schema；② 两段式处理——先做**宽松模式解析**（如 `"123"` → `123`），再做**约束校验**；③ 校验核心用 Rust（pydantic-core）重写，性能比 v1（纯 Python）提升 5~50 倍；④ 失败时抛出结构化的 `ValidationError`，**一次性收集全部错误**而不是只报第一个；⑤ v2 提供全新 API（`model_validate()` / `model_dump()`），替代 v1 的 `.dict()` / `.json()` / `parse_obj()`。

适用范畴：FastAPI 请求/响应模型、配置管理（`pydantic-settings`）、爬虫/ETL 数据清洗，以及任何"外部数据进入系统"的边界。

直观类比：像机场安检——所有行李（数据）必须过同一台扫描仪（Schema），违禁品（非法值）当场没收并给你一张清单（`ValidationError` 列出所有问题），通过后行李被贴上标准标签（类型确定的对象）才能登机（进入系统）。

## 原理
**为什么这样设计**：输入不可信、手写校验啰嗦易漏，因此把"防御脏数据"从手写代码变成声明式注解。校验只在**实例化/反序列化时**发生（创建时一次性把关），这也是它和"事后随手改属性"场景的天然边界。

**核心机制（数据流三步）**：
1. `BaseModel` 子类在**类创建时**读取类型注解，由用 Rust 编写的校验核心（pydantic-core）为每个字段生成一个 Schema；
2. 实例化时数据流经该 Schema：先做**宽松模式解析**（如 `"123"` → `123`、字符串两侧空白去除），再做约束校验（`Field(ge=0, le=150)`、`min_length/max_length`、正则、字段校验器、模型校验器等）；
3. 全部通过 → 得到类型**确定**的对象；任一失败 → 抛出结构化的 `ValidationError`，其中包含**所有**错误（每条的 `type` 与 `loc` 可用于定位和提示），而不是只报第一个。

**关键设计决策**：
- 校验核心用 Rust 重写（v1 是纯 Python），性能提升 5~50 倍，这是 v2 版本号背后最大的架构变化；
- 失败时收集**全部**错误而非快速失败，是为了方便前端一次性向用户提示所有问题；
- v2 更严格：把 `float` 类型的 `1.5` 传给 `int` 字段，v1 会悄悄截断成 `1`，v2 默认**报错**，避免静默丢精度；
- 字段默认值语义：`age: int = 1` 表示"可省略且默认 1"，而**不是**"必填且默认 1"；`Optional[int] = None` 表示"可为 None 且默认 None"；
- 默认是宽松模式（str→int 会自动转），需要 Python 原生严格行为时用 `strict=True`；
- 派生值用 `@computed_field` 在 dump 阶段计算，会出现在 `model_dump()` 结果里（不参与校验输入）。

## 应用
**典型使用场景**：FastAPI 请求/响应模型、配置管理（`pydantic-settings`）、爬虫/ETL 数据清洗，以及任何"外部数据进入系统"的边界——API 入参、JSON 配置、数据库行反序列化。

**快速上手步骤**：
1. `pip install "pydantic>=2"`，定义继承 `BaseModel` 的类，给字段写类型注解和约束（`Field`、校验器、`ConfigDict`）；
2. 用字典实例化 `User(**data)`，或用 v2 新 API `User.model_validate(json_str_or_dict)` 反序列化——自动校验 + 类型转换；
3. 用 `model_dump()` / `model_dump_json()` 序列化；失败时捕获 `ValidationError`，遍历 `e.errors()` 拿到每条错误的 `type` 和 `loc` 做用户提示。

最小示例：`class M(BaseModel): x: int`，然后 `M(x="3").x == 3` 两行即可完成"字符串→int"的自动转换与校验。

**注意事项 / 常见坑**：
- ❌ 还在用 v1 的 API（`@validator`、`@root_validator`、`.dict()`、`.json()`、`parse_obj()`），升级 v2 后报 `DeprecationWarning` 或直接 `AttributeError`。✅ 换成 v2 API：`@field_validator`、`@model_validator(mode='after')`、`model_dump()`、`model_dump_json()`、`model_validate()`；
- ❌ 以为 `age: int = 1` 是"必填且默认 1"——实际是**可省略**（有默认值就不必填）；写 `Optional[int]` 却发现 `None` 被放行。✅ 必填就不给默认值；`Optional[int] = None` 表示"可为 None 且默认 None"；若要求"必须显式传入（哪怕是 None）"，用 `Field(...)` 或 v2.7+ 的 `model_config = ConfigDict(validate_default=True)` 配合校验器；
- ❌ 把 `float` 类型的 `1.5` 传给 `int` 字段，以为会自动截断——v1 会悄悄截断成 `1`，v2 默认**报错**（v2 更严格）。✅ 数据源确实有小数时用 `float`，或在字段上用自定义转换器处理；
- ⚠️ 校验只在**实例化/反序列化时**发生，之后直接改属性（`u.age = -1`）默认**不触发校验**，需要 `model_config = ConfigDict(validate_assignment=True)`；
- ⚠️ 复杂业务规则（如"查数据库判断唯一性"）不适合放校验器里，那是业务层的职责，不要把所有逻辑都塞进模型。

```python
# pip install "pydantic>=2"
from datetime import datetime
from pydantic import (BaseModel, Field, ConfigDict,
                      field_validator, model_validator,
                      computed_field, ValidationError)

class User(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)  # 全局：字符串自动去空白
    id: int
    name: str = Field(min_length=1, max_length=20)        # 约束：长度范围
    email: str
    age: int = Field(ge=0, le=150)                        # 约束：数值范围 0~150
    tags: list[str] = []                                  # 默认值：空列表
    created_at: datetime = Field(default_factory=datetime.now)  # 可变默认值用 factory

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

**案例详解**：① 第 1 段演示**宽松解析**——`id="123"` 自动转成 `int`，`name` 两侧空白被 `str_strip_whitespace` 去掉，`is_adult` 由 `@computed_field` 在 dump 时派生；② 第 2 段演示 v2 反序列化新 API `model_validate()` 与序列化新 API `model_dump()`，替代 v1 的 `.dict()`；③ 第 3 段演示失败场景——`email="bad-email"` 触发字段校验器、`age=200` 触发 `le=150` 上限，`e.errors()` 一次性返回两条错误（`value_error` + `less_than_equal`），每条带 `loc` 字段定位。完整可运行版本见"具体案例"中的 `pydanticv2_sample.py`。

---
## 关联
- 前置：[[Python 类型注解]]、[[dataclass]]
- 类似：[[dataclass]]（区别是 dataclass 只做存储不校验、不转换类型，是"哑"容器）；[[marshmallow]]（区别是 marshmallow 需要为序列化和反序列化各写一套 Schema，Pydantic 一份注解通吃）
- 进阶：[[FastAPI]]（请求/响应模型直接复用 Pydantic 的声明式校验，是它最典型的下游应用）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| Pydantic v2（本文） | 类型注解即 Schema，Rust 核心高性能校验+转换 | API 边界、配置管理、需要 JSON Schema 的场景 |
| dataclasses（标准库） | 零依赖的注解数据容器，无校验 | 内部纯数据传递、不想引入第三方依赖 |
| attrs | 更灵活的类构建器（slots、validators 可选） | 需要精细控制类行为（slots、不可变对象） |
| marshmallow | 显式定义序列化 Schema，校验/序列化分离 | 老项目、需要"加载/转储"两套不同 Schema |

---
## 参考
- [Pydantic v2 官方文档 - Validation](https://docs.pydantic.dev/latest/concepts/validation/)
- [Pydantic v2 官方文档 - Validators](https://docs.pydantic.dev/latest/concepts/validators/)
- [Pydantic v2 官方文档 - Migration Guide (v1→v2)](https://docs.pydantic.dev/latest/migration/)

---
## 具体案例
- [[pydanticv2_sample.py 应用示例]](pydanticv2_sample.py)（本笔记代码的完整可运行版，含宽松解析、model_validate/model_dump、ValidationError 遍历等场景）
