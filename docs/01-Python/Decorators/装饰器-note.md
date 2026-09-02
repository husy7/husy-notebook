---
title: "Python 装饰器"
tags: [Python, 函数式编程, 语法糖]
date: 2026-08-27
---

# Python 装饰器

## 定义

装饰器就是一个"接收函数、返回新函数"的函数，让你在不修改原函数代码的前提下，给它套上一层额外功能（计时、日志、鉴权、缓存等）。它是对函数做"包裹式增强"的高阶函数 + 语法糖工具，一句话概括：装饰器把横切逻辑从业务逻辑中剥离出来，以"包裹"而非"侵入"的方式统一附加到目标函数上。

- 解决什么问题：多个函数需要重复添加同一类横切逻辑（打印日志、计时、缓存、权限校验），若靠复制粘贴，会产生大量重复代码且难以维护、难以统一修改。
- 核心特征：以函数为输入、以（包装后的）新函数为输出；对目标函数完全透明——被装饰函数体内的代码零改动，调用方签名与用法不变。
- 本质定位：Python 语言级的轻量 AOP 实现，把"横切关注点"（cross-cutting concerns）与业务逻辑解耦。
- 适用范畴：任何"为多个函数统一叠加通用行为"的场景——耗时统计、日志、鉴权、结果缓存、重试机制、路由注册等。
- 语法层面：`@decorator` 只是语法糖，等价于 `func = decorator(func)`，因此其成立的前提是 Python 函数是一等公民（可像普通对象一样被传递与返回）。

## 原理

- 一等公民与高阶函数：Python 中函数本身就是对象，可以像普通值一样被赋值、传递、作为返回值；装饰器正是"接收函数、返回函数"的高阶函数，语法糖 `@decorator` 等价于 `func = decorator(func)`——在模块导入阶段执行该赋值，此后调用 `func()` 实际执行的是装饰器返回的包装函数（wrapper）。
- 闭包机制：wrapper 定义在装饰器内部，通过闭包捕获外层变量 `func`，使每次调用 wrapper 时都能回调到原函数，从而在调用前后插入附加逻辑。
- 为什么必须这样设计：横切逻辑与业务逻辑天然应分离；装饰器利用闭包和高阶函数，以"包裹"而非"侵入"的方式扩展行为，符合开闭原则——对扩展开放，对修改关闭。
- 执行时机差异：`@decorator` 这一层（装饰器函数体）在**导入时执行一次**（完成函数替换），而 wrapper 内的包装逻辑在**每次调用时执行**；不要把需要每次运行才生效的代码放在 wrapper 外层。
- 关键细节：`@functools.wraps(func)` 内部通过 `update_wrapper` 把原函数的 `__name__`、`__doc__` 等元信息复制到 wrapper 上，保住函数身份；类装饰器修饰实例方法时涉及描述符协议（descriptor protocol），绑定行为与普通函数不同，处理更复杂；装饰器替换函数后，函数对象身份已改变，涉及 `isinstance` / `id` 比较时可能失效。

## 应用

- 典型场景：接口鉴权、执行耗时统计、结果缓存（`functools.cache`）、重试机制、注册路由（如 Flask 的 `@app.route`）。
- 快速上手（三步）：① 定义装饰器——接收一个函数 `func`，在内部定义 `wrapper(*args, **kwargs)` 并返回它；② 在 `wrapper` 里执行附加逻辑，在调用 `func(*args, **kwargs)` 的前后包裹；③ 用 `@functools.wraps(func)` 装饰 wrapper 保留元信息，再用 `@装饰器名` 应用到目标函数。
- ❌ 坑 1：忘了 `@functools.wraps`，被装饰函数的 `__name__` 变成 `wrapper`、docstring 丢失，调试和反射（如按函数名注册）出错 → ✅ 在 wrapper 定义上方加 `@functools.wraps(func)`。
- ❌ 坑 2：wrapper 没用 `*args, **kwargs` 透传参数，导致装饰器只能匹配固定签名，换函数复用就报 `TypeError` → ✅ wrapper 统一签名 `def wrapper(*args, **kwargs)` 并原样传给 `func`。
- ❌ 坑 3：带参数装饰器少写一层嵌套，`@deco(3)` 时把参数 3 当成了函数对象 → ✅ 带参装饰器是"三层结构"：最外层接收参数、返回真正的装饰器，装饰器再返回 wrapper。
- 边界条件：装饰器在**导入时**执行一次、包装逻辑在**每次调用时**执行，不要在 wrapper 外层放需要每次运行的代码；类装饰器修饰实例方法时需考虑描述符协议（可用 `functools.wraps` 配合方法签名，或直接使用 `functools` 提供的工具类）；装饰器会改变函数身份，涉及 `isinstance` / `id` 比较时可能失效。

```python
# 可运行示例：统计函数执行耗时的装饰器（含逐行注释 + 案例详解）
import functools
import time

def timer(func):
    """装饰器：接收一个函数 func，返回包装后的新函数 wrapper"""
    @functools.wraps(func)          # 关键①：把 func 的 __name__/__doc__ 等元信息复制给 wrapper，
                                    #         否则 slow_add.__name__ 会变成 wrapper
    def wrapper(*args, **kwargs):   # 关键②：统一透传签名，保证任意签名函数都能被本装饰器复用
        start = time.perf_counter()     # 附加逻辑（前）：每次调用时记录起始时间
        result = func(*args, **kwargs)  # 调用原函数，透传全部位置参数与关键字参数
        cost = time.perf_counter() - start  # 附加逻辑（后）：计算实际耗时
        print(f"[timer] {func.__name__} 耗时 {cost:.6f} 秒")
        return result               # 原样返回结果，对调用方保持透明
    return wrapper                  # 返回新函数，替换原函数名

@timer                              # 语法糖：等价于 slow_add = timer(slow_add)，导入时执行一次
def slow_add(a, b):
    time.sleep(0.1)                 # 模拟耗时操作（0.1 秒）
    return a + b

# 案例详解：
# 1. 调用 slow_add(3, 4) 时，实际执行的是 wrapper：先计时 → 再调用真正的 slow_add 完成相加 → 打印耗时。
# 2. slow_add.__name__ 因 @functools.wraps 仍为 "slow_add"；若去掉该装饰器会变成 "wrapper"。
# 3. 该 timer 装饰器可复用于任意函数（只要函数签名用 *args, **kwargs 透传），实现"一处定义、多处复用"。
print("结果 =", slow_add(3, 4))
print("函数名 =", slow_add.__name__)  # 输出 slow_add，而非 wrapper
```

---
## 关联
- 前置：[[Python 一等函数]]、[[Python 闭包]]
- 类似：[[AOP（面向切面编程）]]（区别是 AOP 是框架级的横切注入机制，装饰器是 Python 语言级的轻量实现）；[[上下文管理器]]（区别是装饰器包裹的是函数，`with` 包裹的是代码块，后者作用域更小且必然释放资源）
- 进阶：带参数的装饰器、装饰器类（实现 `__call__`）、`functools.cache` / `functools.lru_cache`、`dataclasses.dataclass` 装饰器原理、多重装饰器的执行顺序（自下而上包裹、自上而下调用）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 装饰器（本文方案） | 用高阶函数包裹原函数，透明叠加横切逻辑 | 需要复用到多个函数上的通用增强（计时、日志、缓存） |
| 手动包装（显式 `func = wrap(func)`） | 直接调用包装函数生成新函数，不使用语法糖 | 仅需一次性包装、不需要语法糖和复用的场景 |

---
## 参考
- [Python 官方文档：装饰器定义](https://docs.python.org/zh-cn/3/glossary.html#term-decorator)
- [Python 官方文档：functools](https://docs.python.org/zh-cn/3/library/functools.html)

---
## 具体案例
- [Python 装饰器计时示例](Python装饰器_sample.py)
