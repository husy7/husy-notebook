# -*- coding: utf-8 -*-
"""
functools —— 案例代码（细分篇）
==============================
覆盖：
  1. partial（偏函数）与 partialmethod
  2. lru_cache（记忆化缓存）
  3. reduce（折叠/累积）
  4. wraps（保留被装饰函数元信息，配合装饰器）
  5. singledispatch（单分派泛型函数）
  6. total_ordering / cmp_to_key 简介
"""

import functools
from functools import partial, lru_cache, reduce, wraps, singledispatch

# =====================================================================
# 一、partial：预填部分参数得到"新函数"
# =====================================================================
def power(base, exp):
    return base ** exp

square = partial(power, exp=2)      # 固定 exp，只留 base
cube   = partial(power, exp=3)
print("[partial] square(5)=", square(5), " cube(3)=", cube(3))

# 给某个函数固定"标准参数"，减少调用处重复
def request(method, url, **kw):
    return f"{method} {url} {kw}"

post = partial(request, "POST")     # 固定 http method
print("[partial] post =", post("/api/user", json=True))

# =====================================================================
# 二、lru_cache：记忆化递归加速（斐波那契指数 → 线性）
# =====================================================================
@lru_cache(maxsize=128)
def fib(n):
    """带缓存的 fib；不加缓存会指数爆炸。"""
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

print("[lru_cache] fib(40) =", fib(40))
print("[lru_cache] cache 命中信息:", fib.cache_info())

# 手动清缓存
fib.cache_clear()

# 对比：不加缓存版本在中 n 就明显卡顿，这里不实际展示以免脚本变慢
def fib_plain(n):                  # 暴力递归，供对比说明
    return n if n < 2 else fib_plain(n - 1) + fib_plain(n - 2)

# =====================================================================
# 三、reduce：把序列"折叠"成一个结果（横向累积）
# =====================================================================
nums = [1, 2, 3, 4, 5]
print("\n[reduce 求和] reduce(add) =", reduce(lambda a, b: a + b, nums))
print("[reduce 阶乘] reduce(mul) =", reduce(lambda a, b: a * b, range(1, 6)))
print("[reduce 找最大] =", reduce(lambda a, b: a if a > b else b, nums))

# reduce 带初值
print("[reduce 初值100 + ...] =", reduce(lambda a, b: a + b, nums, 100))

# =====================================================================
# 四、wraps：装饰器里保护原函数的 __name__ / __doc__
# =====================================================================
def log(fn):
    @wraps(fn)                       # 不用 wraps 则 name/doc 被覆盖
    def inner(*a, **k):
        print(f"  调用 {fn.__name__}")
        return fn(*a, **k)
    return inner

@log
def hello():
    """hello 的 docstring。"""
    return "hi"

print("\n[wraps] hello.__name__ =", hello.__name__)   # 保留原名 'hello'
print("[wraps] hello.__doc__  =", hello.__doc__)

# =====================================================================
# 五、singledispatch：按首参类型分发（替代一堆 if isinstance）
# =====================================================================
@singledispatch
def render(x):
    """默认实现：无法识别的类型。"""
    return f"unknown:{type(x).__name__}:{x}"

@render.register(int)
def _(x: int):
    return f"int:{x}"

@render.register(str)
def _(x: str):
    return f"str:'{x}'"

print("\n[singledispatch] render(5)  =", render(5))
print("[singledispatch] render('hi') =", render("hi"))
print("[singledispatch] render(3.14) =", render(3.14))

# =====================================================================
# 六、total_ordering / cmp_to_key 简介
# =====================================================================
from functools import total_ordering, cmp_to_key

@total_ordering                      # 只要 __lt__ 与 __eq__，自动补全其它比较
class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __eq__(self, o):
        return (self.x, self.y) == (o.x, o.y)
    def __lt__(self, o):
        return (self.x, self.y) < (o.x, o.y)

pts = [Point(3, 1), Point(1, 4), Point(2, 2)]
print("\n[total_ordering] 排序:", [(p.x, p.y) for p in sorted(pts)])

# cmp_to_key：老式 cmp 函数 → 新式 key
def by_len_cmp(a, b):
    return len(a) - len(b)
words = ["bb", "a", "cccc"]
print("[cmp_to_key] 按长排序:", sorted(words, key=cmp_to_key(by_len_cmp)))

# =====================================================================
# 小结
# =====================================================================
# functools：偏函数(partial) / 记忆化(lru_cache) / 折叠(reduce) /
#          装饰器保护(wraps) / 泛型分发(singledispatch) 等高阶工具。
