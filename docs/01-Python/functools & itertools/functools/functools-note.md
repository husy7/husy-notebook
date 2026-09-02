---
title: "functools"
tags: [Python, functools ]
date: 2026-08-27
---

### 模块简介
`functools` 是 Python 标准库中用于高阶函数操作的工具模块，提供了与函数、装饰器、缓存、比较等相关的实用功能。

---

### 1. `partial` – 固定部分参数
**作用**：预先设置函数的部分参数，返回一个新函数。

**语法**：`functools.partial(func, *args, **kwargs)`

**示例**：
```python
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube   = partial(power, exponent=3)

print(square(5))  # 25
print(cube(5))    # 125
```

**适用场景**：需要多次调用同一函数且某些参数固定时，减少重复传参。

---

### 2. `lru_cache` – 缓存函数结果
**作用**：装饰器，缓存函数调用结果（LRU 策略），提高重复计算性能。

**语法**：`@functools.lru_cache(maxsize=128, typed=False)`

**示例**：
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

print(fib(100))  # 极快，中间结果被缓存
```

**适用场景**：递归、动态规划、计算密集型且参数重复率高的纯函数。

---

### 3. `wraps` – 保留原函数元信息
**作用**：装饰器，在自定义装饰器内部使用，将被装饰函数的 `__name__`、`__doc__` 等信息复制到包装函数上。

**示例**：
```python
from functools import wraps

def my_decorator(func):
    @wraps(func)
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

print(say_hello.__name__)  # say_hello
print(say_hello.__doc__)   # 文档字符串
```

**适用场景**：编写装饰器时，保持原函数的元数据，便于调试和文档生成。

---

### 4. `reduce` – 累积运算
**作用**：对序列中的元素从左到右依次应用二元函数，将结果累积为单个值。

**语法**：`functools.reduce(function, iterable[, initializer])`

**示例**：
```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]
total = reduce(lambda x, y: x + y, numbers)
print(total)  # 15

# 带初始值
total_with_initial = reduce(lambda x, y: x + y, numbers, 10)
print(total_with_initial)  # 25
```

**适用场景**：求和、求积、拼接字符串等需要将序列归约为一个值的操作。

---

### 5. `total_ordering` – 自动补全比较方法
**作用**：类装饰器，只需定义 `__eq__` 和一个其他比较方法（如 `__lt__`），自动补全所有比较方法。

**示例**：
```python
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
print(a > b)   # True（自动生成 __gt__）
print(a >= b)  # True（自动生成 __ge__）
```

**适用场景**：自定义类需要完整的比较操作，但不想手动编写所有比较方法。

---

### 6. `singledispatch` – 单分派泛型函数
**作用**：根据第一个参数的类型自动分派到对应的实现，实现类似函数重载的功能。

**语法**：
```python
@singledispatch
def func(arg):
    ...

@func.register(type)
def _(arg):
    ...
```

**示例**：
```python
from functools import singledispatch

@singledispatch
def process(arg):
    print("默认处理:", arg)

@process.register(int)
def _(arg):
    print("处理整数:", arg)

@process.register(str)
def _(arg):
    print("处理字符串:", arg)

process(10)      # 处理整数: 10
process("hello") # 处理字符串: hello
process(3.14)    # 默认处理: 3.14
```

**适用场景**：需要针对不同类型参数执行不同逻辑，但又希望保持统一的函数接口。

---

### 总结
- `partial`：固定部分参数，简化调用。
- `lru_cache`：缓存计算结果，提升性能。
- `wraps`：保留装饰器中原函数的元信息。
- `reduce`：序列累积运算，函数式编程利器。
- `total_ordering`：自动生成比较方法，减少样板代码。
- `singledispatch`：根据第一个参数类型分派实现，模拟函数重载。

这些工具可以写出更简洁、高效、可维护的 Python 代码。