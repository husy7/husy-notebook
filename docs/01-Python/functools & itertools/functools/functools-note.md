---
title: "functools"
tags: [Python, functools]
date: 2026-08-27
---

# functools

## 定义
`functools` 是 Python 标准库中用于"高阶函数（higher-order functions）与可调用对象"操作的工具模块，官方将其归类在函数式编程模块（Functional Programming Modules）中，与 `itertools` 并列。

它解决的核心问题是：函数式编程与装饰器开发中反复出现的样板代码没有标准答案，手写容易出错且难维护。例如——多次调用同一函数时重复传参、递归/计算密集函数的重复计算、自定义装饰器丢失被装饰函数的元信息、没有内置函数可用时的序列归约、自定义类需要补齐全部比较运算符、希望按参数类型走不同实现却又要保持统一接口。

核心特征：模块内几乎全部是可复用的函数与装饰器工厂：`partial`（固定部分参数生成偏函数）、`lru_cache`/`cache`（带 LRU 策略的记忆化缓存装饰器）、`wraps`/`update_wrapper`（装饰器元信息保留）、`reduce`（左折叠归约）、`total_ordering`（自动补全富比较方法）、`singledispatch`/`singledispatchmethod`（按第一个参数类型分派的泛型函数）。

适用范畴：函数式编程、装饰器编写、记忆化性能优化、类型多态分派、对象排序比较等几乎所有"把函数当值处理"的场景。

这些工具的共同设计取向：把经过充分测试的通用逻辑封装成标准库能力，让调用方用最少的样板代码写出更简洁、高效、可维护的 Python 代码。

## 原理
`functools` 各工具的核心机制如下：

- `partial(func, *args, **kwargs)` 的本质是闭包/包装对象：保存 `func` 与已固定的位置/关键字参数，调用返回的新函数时把"固定参数 + 本次传入参数"合并后转发给 `func`，等价于 `lambda **kw: func(*固定args, **kw)`，且保留 `func`/`args`/`keywords` 属性便于自省。`square(5)` 实际执行 `power(5, exponent=2)`。

- `lru_cache` 是记忆化（memoization）装饰器：把调用参数整理成可哈希键，将返回值存入字典；同时用双向链表维护"最近最少使用"顺序——每次命中把条目移到最近端，条目数超过 `maxsize` 时淘汰最久未使用的一端。`typed=True` 时 `1` 与 `1.0` 视为不同键；`maxsize=None` 时退化为不淘汰的无限缓存。由于以参数哈希为键，**参数必须可哈希**；`fib` 例子中每个 `n` 只计算一次，时间复杂度从指数级降为 O(n)。

- `wraps` 内部调用 `update_wrapper(wrapper, wrapped)`：把原函数的 `__name__`、`__qualname__`、`__doc__`、`__module__`、`__annotations__`、`__dict__` 等拷贝/更新到包装函数，并设置 `wrapper.__wrapped__ = func`，使 `help()`、`inspect.signature`、自动文档工具以及多层装饰时都能追踪到最内层的原始函数。

- `reduce(function, iterable[, initializer])` 执行左折叠（left fold）：取前两个元素应用函数得中间值，再与下一个元素应用函数……直到序列耗尽，等价于 `f(f(f(x1, x2), x3), ...)`；提供 `initializer` 时先与第一个元素结合；空序列且无 `initializer` 时抛 `TypeError`。

- `total_ordering` 是类装饰器：只要类定义了 `__eq__` 与其余六个比较运算符中的任意一个（惯例是 `__lt__`），它就按比较运算间的逻辑关系补出缺失方法，如 `a > b` 等价于 `b < a`、`a >= b` 等价于 `not (a < b)`、`a != b` 等价于 `not (a == b)`。

- `singledispatch` 把一个普通函数变成泛型函数：内部维护 `{类型: 实现}` 注册表（基函数作为默认实现）。调用时取**第一个实参的类型**查表；未精确命中时沿该类型的 `__mro__`（方法解析顺序，含 `collections.abc` 抽象基类的注册关系）寻找最近的已注册父类实现；都没有才回退到基函数。因此它只做"单分派"——仅按第一个参数类型选择实现，这正是与函数重载（overload）的本质差异。

## 应用
典型使用场景与快速上手：

- **`partial`**：预置配置的回调、事件处理器、GUI 按钮绑定、给 `sorted`/`map` 传带固定选项的函数等。上手：`new_func = partial(old_func, 固定参数)`，之后用更少参数调用即可。坑：以关键字固定参数后，调用时不能再以同名关键字传参，否则 `TypeError: got multiple values for argument`；在"只需要一个可调用对象"的位置（如回调）临时造函数，比定义具名函数或写 lambda 更清晰。

- **`lru_cache`**：递归与动态规划（fib、爬楼梯）、解析/校验类纯函数、以相同参数反复调用的计算密集型逻辑。坑：① 参数必须可哈希（`list`/`dict` 不行）；② 被缓存函数应保持纯函数，若带副作用，命中缓存会静默跳过执行；③ 装饰在实例方法上时 `self` 也是键的一部分，导致实例被缓存长期持有、无法被垃圾回收（内存泄漏风险），应优先设计成无状态顶层函数或使用 `weakref`；④ `maxsize=None` 表示无限缓存，Python 3.9+ 无大小限制时可直接用 `@functools.cache`。

- **`wraps`**：编写任何自定义装饰器都应在内层 wrapper 上加 `@wraps(func)`。坑：忘记加时 `say_hello.__name__` 会变成 `wrapper`、`__doc__` 丢失，调试日志与自动文档全部错乱。

- **`reduce`**：对序列做"从左到右累积成单值"且没有内置函数时的通用手段，如求积、字符串拼接、构造字典、自定义折叠。坑：纯求和优先用内置 `sum()`（更快更清晰）；Python 3 中 `reduce` 不再是内置函数，必须显式 `from functools import reduce`；逻辑复杂时 for 循环可读性往往更好，勿过度使用。

- **`total_ordering`**：自定义类需要完整比较（`sorted`/`min`/`max` 与 `==`/`<`/`>=` 等）但不想手写六个方法。上手：定义 `__eq__` + `__lt__` 后加 `@total_ordering`。坑：它只是"样板推导"，每个比较仍是独立方法调用，大规模频繁比较有额外性能开销；比较逻辑必须自洽；能用 `@dataclass(order=True)` 的场合优先用它。

- **`singledispatch`**："统一函数接口 + 按类型分别实现"，适合多类型序列化/格式化、参数校验、事件处理、分类型打印等。上手：基函数加 `@singledispatch`，再 `@process.register(类型)` 链式注册各类型实现（支持 `tuple`/多个连续注册、注册 ABC 或自定义基类，子类实例自动落到父类实现）。坑：只按第一个参数分派；`@process.register` 后各实现建议用 `_` 命名避免覆盖；类内方法场景用 `singledispatchmethod`（Python 3.8+）。

```python
# ========== 1. partial —— 固定部分参数 ==========
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)   # 预先固定 exponent=2
cube   = partial(power, exponent=3)   # 预先固定 exponent=3

print(square(5))  # 25
print(cube(5))    # 125

# 坑：关键字参数已被固定后，调用时不能再传同名关键字：
# square(5, exponent=2)  # TypeError: power() got multiple values for argument 'exponent'

# ========== 2. lru_cache —— 缓存函数结果（LRU 策略） ==========
from functools import lru_cache

@lru_cache(maxsize=128)   # maxsize=None 表示不限制大小
def fib(n):
    """斐波那契：递归 + 记忆化，避免指数级重复计算"""
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

print(fib(100))  # 极快，中间结果被缓存 → 354224848179261915075

# 坑：参数必须可哈希；函数应保持纯函数（无副作用），否则命中缓存会跳过执行；
#     装饰在实例方法上会持有 self，导致对象无法被回收（内存泄漏风险）。

# ========== 3. wraps —— 保留原函数元信息 ==========
from functools import wraps

def my_decorator(func):
    @wraps(func)   # 把 __name__/__doc__ 等元信息复制到 wrapper 上
    def wrapper(*args, **kwargs):
        print("调用前")
        result = func(*args, **kwargs)
        print("调用后")
        return result
    return wrapper

@my_decorator
def say_hello():
    """文档字符串"""
    print("Hello!")

print(say_hello.__name__)  # say_hello（不加 wraps 会输出 wrapper）
print(say_hello.__doc__)   # 文档字符串（不加 wraps 会输出 None）

# ========== 4. reduce —— 累积运算（左折叠） ==========
from functools import reduce

numbers = [1, 2, 3, 4, 5]
total = reduce(lambda x, y: x + y, numbers)
print(total)  # 15

# 带初始值：10 先与第一个元素结合
total_with_initial = reduce(lambda x, y: x + y, numbers, 10)
print(total_with_initial)  # 25

# 坑：纯求和用内置 sum() 更快更清晰；reduce 适合求积、构造字典等无内置的累积；
#     空序列且无 initializer 会抛 TypeError。

# ========== 5. total_ordering —— 自动补全比较方法 ==========
from functools import total_ordering

@total_ordering
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

    def __eq__(self, other):
        return self.grade == other.grade

    def __lt__(self, other):
        return self.grade < other.grade

a = Student("Alice", 90)
b = Student("Bob", 85)
print(a > b)   # True（__gt__ 自动生成）
print(a >= b)  # True（__ge__ 自动生成）

# 坑：__eq__ + __lt__ 即可让 sorted/max/min 工作，但每个比较仍是独立方法调用，
#     频繁大规模比较存在额外开销。

# ========== 6. singledispatch —— 单分派泛型函数 ==========
from functools import singledispatch

@singledispatch
def process(arg):
    print("默认处理:", arg)   # 基函数 = 兜底实现

@process.register(int)
def _(arg):                   # 注册 int 专用实现
    print("处理整数:", arg)

@process.register(str)
def _(arg):                   # 注册 str 专用实现
    print("处理字符串:", arg)

process(10)      # 处理整数: 10
process("hello") # 处理字符串: hello
process(3.14)    # 默认处理: 3.14（float 未注册，落到基函数）

# 进阶：可注册 ABC/自定义基类（子类实例沿 __mro__ 自动落到父类实现）；
#       也可一次注册多个类型（连续多次 @process.register(tuple) 等）。
```

---
## 关联
- 前置：[[Python 装饰器]]
- 类似：[[itertools]]（区别是 functools 面向"函数/可调用对象"，提供偏函数、缓存、归约、单分派等高阶工具；itertools 面向"可迭代对象"，提供组合、过滤、无限序列等惰性迭代工具；两者常配合用于函数式编程）
- 进阶：[[functools.cached_property 与描述符协议]]

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| functools（本文方案） | 用标准库封装好的高阶工具替代样板代码：`partial` 固定参数、`lru_cache` 记忆化、`wraps` 保留元数据、`reduce` 归约、`total_ordering` 补全比较、`singledispatch` 按首参类型分派 | 追求简洁、正确、低维护成本的标准库场景：装饰器开发、递归优化、回调预置、类型多态、自定义类比较 |
| 手写实现 / 内置替代 | 用闭包、dict 缓存、显式循环、手写六个比较方法或 `if/elif` 类型判断自行实现等价逻辑；能用 `sum()`/`max()` 等内置时不用 `reduce` | 对性能有极致要求或行为需高度定制；仅做简单求和/最值（直接用内置）；比较逻辑依赖具体业务、需完全掌控时 |

---
## 参考
- [functools — 高阶函数和可调用对象上的操作（Python 官方文档）](https://docs.python.org/3/library/functools.html)

---
## 具体案例
- [[functools 实战示例]](functools_sample.py)
