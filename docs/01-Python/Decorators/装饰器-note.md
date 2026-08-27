---
title: "Python 装饰器"
tags: [Python, 函数式编程, 语法糖]
date: 2026-08-27
---
# Python 装饰器
> 装饰器就是一个“接收函数、返回新函数”的函数，让你在不修改原函数代码的前提下，给它套上一层额外功能（如计时、日志、鉴权）。
## 原理 / 动机
- 解决什么实际问题：多个函数需要重复添加同一类横切逻辑（打印日志、计时、缓存、权限校验），复制粘贴会导致大量重复代码且难以维护。
- 核心原理（简洁全面）：Python 中函数是一等公民，可以像普通对象一样被传递和返回。`@decorator` 语法糖等价于 `func = decorator(func)`：装饰器接收原函数，返回一个包装函数（wrapper），调用时实际执行的是包装函数。
- 为什么必须这样设计：横切逻辑与业务逻辑天然应分离，装饰器利用闭包和高阶函数，以“包裹”而非“侵入”的方式扩展行为，符合开闭原则——对扩展开放，对修改关闭。
## 应用示例
- 适用场景：接口鉴权、执行耗时统计、结果缓存（`functools.cache`）、重试机制、注册路由（如 Flask 的 `@app.route`）。
- 快速上手：
  1. 定义装饰器：接收一个函数 `func`，在内部定义 `wrapper(*args, **kwargs)` 并返回它。
  2. 在 `wrapper` 里执行附加逻辑，前后包裹对 `func(*args, **kwargs)` 的调用。
  3. 用 `@functools.wraps(func)` 装饰 `wrapper` 保留元信息，再用 `@装饰器名` 应用到目标函数。
```python
# 可运行示例
import functools
import time
def timer(func):
    """统计函数执行耗时的装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        cost = time.perf_counter() - start
        print(f"[timer] {func.__name__} 耗时 {cost:.6f} 秒")
        return result
    return wrapper
@timer
def slow_add(a, b):
    time.sleep(0.1)
    return a + b
print("结果 =", slow_add(3, 4))
print("函数名 =", slow_add.__name__)  # 输出 slow_add，而非 wrapper
```
## 边界 / 常见坑
- ❌ 错误现象：忘了 `@functools.wraps`，被装饰函数的 `__name__` 变成 `wrapper`，文档字符串丢失，调试和反射（如按函数名注册）出错。
  ✅ 正确做法：在 `wrapper` 定义上方加 `@functools.wraps(func)`。
- ❌ 错误现象：wrapper 没用 `*args, **kwargs` 透传参数，导致函数只能匹配固定签名，换函数复用装饰器就报 `TypeError`。
  ✅ 正确做法：wrapper 统一签名 `def wrapper(*args, **kwargs)` 并原样传给 `func`。
- ❌ 错误现象：带参数装饰器少写一层嵌套，`@deco(3)` 时把参数当成了函数对象。
  ✅ 正确做法：带参装饰器是“三层结构”——最外层接收参数，返回真正的装饰器，装饰器再返回 wrapper。
- 边界条件：装饰器在**导入时**执行一次，包装逻辑在**每次调用时**执行，不要在 wrapper 外层放需要每次运行的代码；类装饰器修饰实例方法时需考虑描述符协议（可用 `functools.wraps` 配合方法签名，或直接使用 `functools` 提供的工具类）；装饰器会改变函数身份，涉及 `isinstance` / `id` 比较时可能失效。
## 关联
- 前置知识：[[Python 一等函数]]、[[Python 闭包]]
- 类似概念：[[AOP（面向切面编程）]]（区别是 AOP 是框架级的横切注入机制，装饰器是 Python 语言级的轻量实现）；[[上下文管理器]]（区别是装饰器包裹的是函数，`with` 包裹的是代码块，后者作用域更小且必然释放资源）。
- 进阶知识：带参数的装饰器、装饰器类（实现 `__call__`）、`functools.cache` / `functools.lru_cache`、`dataclasses.dataclass` 装饰器原理、多重装饰器的执行顺序（自下而上包裹、自上而下调用）。
## 自我检验
1. 我能否一句话说清它解决什么问题？
   - 答：不修改原函数代码，为函数统一叠加横切逻辑（日志、计时、鉴权等）。
2. 我能否写出最小可用示例？
   - 答：定义 `wrapper` 透传 `*args, **kwargs`，调用 `func` 前后执行附加逻辑，返回 wrapper，用 `@` 应用。
3. 我能否说出一个常见错误或边界？
   - 答：不加 `@functools.wraps` 导致函数元信息（`__name__`、docstring）丢失。
4. 我能否说出它和我已会的某个概念的区别？
   - 答：与闭包的区别：闭包是“内层函数引用外层变量”的机制，装饰器是利用闭包实现的一种“替换函数”的用法。
## 对比与选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 装饰器 | 用高阶函数包裹原函数，透明叠加横切逻辑 | 需要复用到多个函数上的通用增强（计时、日志、缓存） |
| 手动包装（显式 `func = wrap(func)`） | 直接调用包装函数生成新函数 | 仅需一次性包装、不需要语法糖和复用的场景 |
## 参考
- [Python 官方文档：装饰器定义](https://docs.python.org/zh-cn/3/glossary.html#term-decorator)
- [Python 官方文档：functools](https://docs.python.org/zh-cn/3/library/functools.html)
## 具体案例
- [Python 装饰器计时示例](Python装饰器_sample.py)
