# -*- coding: utf-8 -*-
"""
迭代器 iterator —— 案例代码（细分篇）
===================================
覆盖：
  1. 可迭代(iterable) vs 迭代器(iterator) 的区别
  2. for 底层循环( iter / next / StopIteration)
  3. 手写自定义迭代器类
  4. 常见的"既是可迭代又是迭代器"行为
  5. 配合内建 zip/map/enumerate/filter（它们也是迭代器）
"""

# =====================================================================
# 一、理解 for 的底层机制：iter + next + StopIteration
# =====================================================================
lst = [10, 20, 30]
# for x in lst 等价于下面手工循环:
it = iter(lst)                     # 1) 拿到迭代器
while True:
    try:
        x = next(it)               # 2) 逐个取
        print("[for底层] x =", x)
    except StopIteration:          # 3) 到头结束
        break

# =====================================================================
# 二、可迭代对象 vs 迭代器：用实际内建验证
# =====================================================================
print("\nlist 是可迭代否有 __iter__? ", hasattr(lst, "__iter__"),
      " 有 __next__(即迭代器)吗? ", hasattr(lst, "__next__"))
i = iter(lst)                       # list 的迭代器
print("迭代器 有 __iter__? ", hasattr(i, "__iter__"),
      " 有 __next__? ", hasattr(i, "__next__"))

# list 可以反复遍历（因为每次 for 都会新建迭代器）:
print("list 可两次 for:", [x for x in lst], [x for x in lst])

# =====================================================================
# 三、手写一个自定义迭代器类：倒计时
# =====================================================================
class Countdown:
    """从 n 倒数到 1。演示 __iter__ + __next__ 的完整协议。"""
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        return self                  # 自己就是迭代器
    def __next__(self):
        if self.n <= 0:
            raise StopIteration      # 必须 raise，否则 for 不结束
        v = self.n
        self.n -= 1
        return v

print("\n[自定义迭代器] 手动 next:")
c = Countdown(3)
print(" ", next(c), next(c), next(c))
try:
    next(c)
except StopIteration:
    print("  (耗尽，正常抛 StopIteration)")

# 用 for 遍历自动处理结束：
print("[自定义迭代器] for 遍历:", [x for x in Countdown(5)])

# =====================================================================
# 四、一次性陷阱：迭代器(与文件/生成器)只能消费一次
# =====================================================================
it_once = iter([1, 2, 3, 4])
print("\n[迭代器一次性] 第一次:", list(it_once))
print("[迭代器一次性] 第二次为空:", list(it_once),
      "← 迭代器已耗尽，不能再遍历")

# next(it, default)：给默认值就不抛异常
it_default = iter([5])
print("[next默认] 取完5再用默认:", next(it_default, None),
      "→", next(it_default, "fallback"))

# =====================================================================
# 五、map / filter / enumerate / zip 都返回迭代器（惰性）
# =====================================================================
nums = [1, 2, 3, 4, 5]
# map
added = map(lambda x: x + 10, nums)
print("\n[map] 遍历:", list(added), "再遍历空:", list(added), "(一次性)")

# filter: 惰性产出满足条件的元素
evens = filter(lambda x: x % 2 == 0, nums)
print("[filter] 偶数:", list(evens))

# enumerate: 产出 (索引, 值)
for idx, val in enumerate(["a", "b", "c"], start=1):
    print("  ", idx, val)

# =====================================================================
# 小结
# =====================================================================
# 迭代器核心 = __next__ 每次都出一个值，耗尽 raise StopIteration；
# 可迭代 ≠ 迭代器（list 可重复 for，迭代器一次性）。
