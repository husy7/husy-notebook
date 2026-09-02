# -*- coding: utf-8 -*-
"""
生成器 generator —— 案例代码（细分篇）
=================================
覆盖：
  1. 生成器函数与 yield 机制（惰性产出）
  2. 生成器表达式 vs 列表推导（内存对比）
  3. next / for / yield from
  4. 一次性与可永久重新可迭代的区别
  5. generator 与迭代协议的关联
"""

# =====================================================================
# 一、生成器函数：暂停/恢复的核心机制
# =====================================================================
def count_up(n):
    """生成 0..n-1。每次 next 暂停在 yield，下次从此恢复。"""
    i = 0
    while i < n:
        yield i          # 产出 i 并暂停
        i += 1

g = count_up(3)
print("[生成器对象]", g, type(g))          # <generator object ...>
print("[next]1", next(g))                   # 0
print("[next]2", next(g))                   # 1
print("[next]3", next(g))                   # 2
try:
    next(g)                                  # 耗尽 → StopIteration
except StopIteration:
    print("[StopIteration] 生成器耗尽")

# =====================================================================
# 二、惰性 vs 急切：生成器表达式与列表推导
# =====================================================================
gen_sq   = (x * x for x in range(10))      # 生成器表达式（惰性）
list_sq  = [x * x for x in range(10)]      # 列表推导（急切、一次性建出）

print("\n[生成器表达式] 是一次性累加的:", sum(gen_sq))
print("[列表推导] 结果:", list_sq, "可重复用/支持下标:", list_sq[3])

# 内存差异：大范围更明显（这里只演示对象类型）
import sys
small_list = [i for i in range(1000)]
small_gen  = (i for i in range(1000))
print("\n[内存] 列表推导 sys.getsizeof ≈", sys.getsizeof(small_list),
      "| 生成器 sys.getsizeof ≈", sys.getsizeof(small_gen))

# =====================================================================
# 三、for 循环消费 + yield from 委派
# =====================================================================
def flatten(nested):
    """把嵌套可迭代 flat（延迟、逐项）摊平。"""
    for it in nested:
        yield from it                # 逐个转发子迭代的元素

top = [[1, 2], [3], [4, 5]]
print("\n[yield from] flatten(", top, ") =", list(flatten(top)))

# =====================================================================
# 四、一次性陷阱：generator 消耗后不可重来
# =====================================================================
one = (i for i in range(3))
print("\n[一次性] 第一次 for 消耗:", list(one))
print("[一次性] 第二次 for 为空:", list(one), "← 生成器已耗尽")

# 需要"可重复用"就用 itertools.tee 或改成列表
from itertools import tee
again_src, again_src2 = tee((i for i in range(3)), 2)   # 克隆两个
print("[tee] 克隆后都可独立消费:", list(again_src), list(again_src2))

# =====================================================================
# 五、生成器与无限序列：极佳使用场景
# =====================================================================
def primes_upto_first_n(n):
    """产出前 n 个素数（可无限延展的流式写法只生必要个数）。"""
    num, count = 2, 0
    while count < n:
        if all(num % d for d in range(2, int(num**0.5) + 1)):
            yield num
            count += 1
        num += 1

first_ten = primes_upto_first_n(10)
print("\n[流式素数生成器] 前 10 个素数:", list(first_ten))
# 用 next 一点点取，而非一次全算，适合"只要前几个就够"的场景

# =====================================================================
# 小结
# =====================================================================
# 生成器 = yield 写的惰性迭代器，内存友好、可表达无限/大数据流；
# 但一次性、不可下标/长度，需要持久需 list/tee。
