
# 实战案例：电商支付回调处理系统
#**业务背景**：你的服务接收微信/支付宝的支付回调，处理用户订单。数据来源不可信、渠道格式不同、嵌套层级深——Pydantic v2 的所有核心能力都用得上。
## 完整代码（已实测）

from typing import Annotated, Generic, TypeVar, Literal, Union
from datetime import datetime
from decimal import Decimal
from pydantic import (BaseModel, Field, ConfigDict, field_validator,
                      model_validator, computed_field, StringConstraints,
                      TypeAdapter, ValidationError, BeforeValidator)
T = TypeVar('T')
# ═══════════ ① 自定义可复用类型（Annotated = 约束跟着类型走）═══════════
Money = Annotated[Decimal, Field(gt=0, decimal_places=2)]   # 金额：正数、最多2位小数
Phone = Annotated[str, StringConstraints(pattern=r"^1\d{10}$")]  # 手机号正则
def strip_lower(v: str) -> str:
    return v.strip().lower()                                # 预处理：去空格+转小写
Email = Annotated[str, BeforeValidator(strip_lower),
                  StringConstraints(pattern=r"^[\w.+-]+@[\w-]+\.[\w.]+$")]
# ═══════════ ② 嵌套模型 + 拒绝未知字段 ═══════════
class Address(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True,
                              extra="forbid")  # 传多余字段直接报错（防 is_admin 攻击）
    province: str
    city: str
    detail: str = Field(min_length=5, max_length=100)
# ═══════════ ③ 用户模型：字段名别名 + 计算字段 ═══════════
class User(BaseModel):
    model_config = ConfigDict(populate_by_name=True)  # 别名/原名都能传
    user_id: int
    name: str = Field(min_length=1)
    email: Email            # 脏数据 " Alice@Example.COM " → "alice@example.com"
    phone: Phone
    vip_level: int = Field(default=0, ge=0, le=9)
    addr: Address
    @computed_field        # 序列化时自动出现，不用手动算
    @property
    def masked_phone(self) -> str:
        return f"{self.phone[:3]}****{self.phone[-4:]}"   # 138****8000
# ═══════════ ④ 泛型模型：一个分页类包装任何类型 ═══════════
class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int = Field(ge=0)
    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=100)
    @computed_field
    @property
    def has_more(self) -> bool:
        return self.page * self.size < self.total
# ═══════════ ⑤ 判别联合：不同渠道回调，结构完全不同 ═══════════
class WechatPayCallback(BaseModel):
    channel: Literal["wechat"] = "wechat"
    appid: str
    mchid: str
    amount: Money
    out_trade_no: str
class AlipayCallback(BaseModel):
    channel: Literal["alipay"] = "alipay"
    buyer_id: str
    total_amount: Money
    trade_no: str
    gmt_create: datetime     # 字符串 "2026-08-27 10:00:00" 自动转 datetime
Callback = Annotated[Union[WechatPayCallback, AlipayCallback],
                     Field(discriminator="channel")]  # 按 channel 字段自动路由！
# ═══════════ ⑥ 订单：串联一切 + 三种校验器 ═══════════
class Order(BaseModel):
    model_config = ConfigDict(validate_assignment=True)  # 事后改属性也校验！
    order_id: str = Field(pattern=r"^ORD\d{10}$")
    user: User
    items: list[tuple[str, int, Money]]   # 元组也能校验：(名称, 数量, 单价)
    callback: Callback
    coupon: float | None = None           # Python 3.10+ 新语法 = Optional
    @field_validator("items")             # 单字段跨元素校验
    @classmethod
    def check_items(cls, v):
        if not v:
            raise ValueError("订单至少要有 1 个商品")
        for name, qty, _ in v:
            if qty <= 0:
                raise ValueError(f"商品 {name} 数量必须>0")
        return v
    @model_validator(mode="after")        # 跨字段校验（能访问所有字段）
    def check_coupon(self):
        if self.coupon and self.coupon >= self.total:
            raise ValueError("优惠券金额不能超过订单总额")
        return self
    @computed_field
    @property
    def total(self) -> Decimal:
        return sum(qty * price for _, qty, price in self.items)

## 使用演示（实际输出）

# ---- 场景1：微信回调进来，全是字符串脏数据 ----
wechat_payload = {
    "order_id": "ORD2026082701",
    "user": {"user_id": 1, "name": "Alice",
             "email": "  Alice@Example.COM ",        # ← 脏：空格+大写
             "phone": "13800138000",
             "addr": {"province": "浙江", "city": "杭州", "detail": "西湖区文一西路96号"}},
    "items": [("机械键盘", 1, "399.00"), ("鼠标垫", 2, "29.90")],
    "callback": {"channel": "wechat", "appid": "wx123", "mchid": "m456",
                 "amount": "458.80", "out_trade_no": "T001"},
}
order = Order.model_validate(wechat_payload)
print(type(order.callback).__name__)   # WechatPayCallback ← 自动路由到正确模型
print(order.user.email)                # alice@example.com   ← 自动清洗
print(order.user.masked_phone)         # 138****8000         ← 计算字段
print(order.total)                     # 458.80              ← Decimal 精确计算
# ---- 场景2：支付宝回调，同一个 Order 类直接复用 ----
alipay_payload = {**wechat_payload, "order_id": "ORD2026082702",
    "callback": {"channel": "alipay", "buyer_id": "b1",
                 "total_amount": "458.80", "trade_no": "T002",
                 "gmt_create": "2026-08-27 10:00:00"}}
alipay = Order.model_validate(alipay_payload)
print(alipay.callback.gmt_create)      # 2026-08-27 10:00:00 ← str 自动转 datetime
# ---- 场景3：泛型分页，一套代码包装任何模型 ----
Page[User](items=[order.user], total=100, page=1, size=20).has_more  # True
# ---- 场景4：TypeAdapter 校验裸数据（无需建模型）----
TypeAdapter(list[Money]).validate_python(["1.00", "2.50"])
# → [Decimal('1.00'), Decimal('2.50')]
# ---- 场景5：拦截各种脏数据 ----
try:
    order.order_id = "bad-id"          # 事后修改（validate_assignment 生效）
except ValidationError: ...            # ❌ string_pattern_mismatch
try:
    Order.model_validate({**wechat_payload, "coupon": 999})  # 优惠券>总额
except ValidationError as e: ...       # ❌ 优惠券金额不能超过订单总额

## Pydantic v2 特性覆盖清单
# | # | 特性 | 案例中的位置 | 解决的问题 |
# |---|------|------------|-----------|
# | 1 | `Field()` 约束 | `gt/le/min_length/pattern/decimal_places` | 语义脏数据（负金额、超量购买） |
# | 2 | `ConfigDict` | `extra="forbid"` / `str_strip_whitespace` / `validate_assignment` | 未知字段攻击 / 自动清洗 / 事后修改校验 |
# | 3 | `Annotated` 自定义类型 | `Money` / `Phone` / `Email` | 约束跟着类型走，一处定义处处复用 |
# | 4 | `BeforeValidator` | `Email` 的 `strip_lower` | 校验前先清洗数据 |
# | 5 | `field_validator` | `check_items` | 单字段复杂校验 |
# | 6 | `model_validator(mode='after')` | `check_coupon` | **跨字段**校验（优惠券 vs 总额） |
# | 7 | `computed_field` | `total` / `masked_phone` / `has_more` | 校验后派生值，序列化时自动带上 |
# | 8 | 嵌套模型 | `Order.user.addr` | 深层结构逐层校验，错误带精确路径 |
# | 9 | `Literal` + **判别联合** | `Callback` + `discriminator="channel"` | 一套代码处理多渠道不同结构（O(1) 路由） |
# | 10 | 泛型模型 | `Page[T]` | 一个分页类包装所有模型 |
# | 11 | `TypeAdapter` | `prices` | 不建模直接校验裸数据 |
# | 12 | `model_validate` / `model_dump_json` | 入口/出口 | 字符串脏数据 → 类型安全对象 → 精确序列化 |
# | 13 | 宽松模式类型转换 | `"399.00"`→`Decimal`，`"2026-08-27 10:00:00"`→`datetime` | JSON 全是字符串的问题 |
# | 14 | `ValidationError` | 场景5 | 一次性收集全部错误，含类型/位置/原始输入 |
# | 15 | Python 3.10+ 语法 | `float \| None`、`list[str]` | 现代注解风格 |
## 为什么这个案例值得抄走
# 1. **判别联合是精髓**（#9）：真实系统必然有"同一个入口、多种消息格式"——不同支付渠道、不同通知类型、不同 webhook。`discriminator` 让 Pydantic 按 `channel` 字段直接路由到对应模型，比手写 `if channel == "wechat"` 优雅且快（不用逐个尝试解析）。
# 2. **`Money = Annotated[Decimal, ...]` 是工程最佳实践**：金额约束只定义一次，全项目 `Order`、`Callback`、退款单全部复用。
# 3. **错误处理天然结构化**：`e.errors()` 每条都带 `loc`（精确到嵌套路径如 `('user', 'addr', 'detail')`）、`type`（机器可读）、`input`（原始脏值）——直接喂给日志或返回给前端。
# **可扩展方向**（进阶特性，需要时再加）：
# - `pydantic-settings`：把环境变量/`.env` 文件直接变成带校验的配置对象
# - `model_copy(update={...})`：不可变风格的"修改"
# - `TypeAdapter(...).dump_json()` + `model_dump(mode="json")`：序列化为 JSON 兼容类型
# - 自定义 `@model_serializer` / `@field_serializer`：控制序列化格式（如日期输出 `YYYY/MM/DD`）
# - `Strict` 模式：安全敏感接口要求类型严格匹配

