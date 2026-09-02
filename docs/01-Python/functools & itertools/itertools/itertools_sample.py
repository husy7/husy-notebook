# -*- coding: utf-8 -*-
"""
itertools —— 案例代码（细分篇）
===============================
覆盖：
  1. chain / islice / groupby / tee / accumulate
  2. count / cycle / repeat（无限迭代与安全截断）
  3. product / permutations / combinations（小心别 list 全量）
  4. 惰性：验证它们吃的都是"迭代器"而不是一次性塞内存
"""

import itertools as it

# =====================================================================
# 一、chain：把多个来源串成一条流
# =====================================================================
c = it.chain([1, 2], (3, 4), "ab")
print("[chain]", list(c))          # [1, 2, 3, 4, 'a', 'b']

# 摊平一个二维列表（列表生成器版手动 vs chain）
nested = [[1, 2], [3], [4, 5]]
print("[chain.from_iterable]", list(it.chain.from_iterable(nested)))

# =====================================================================
# 二、islice：对迭代器做“流式切片”（无法用下标）
# =====================================================================
# 对一个“看起来无限”的范围只取其中一小段——省内存
seq = it.islice(range(1_000_000), 5, 11)
print("[islice]", list(seq))        # [5, 6, 7, 8, 9, 10]

# =====================================================================
# 三、groupby：相邻且键相同的项合并成组（先 sort 才能全局聚合）
# =====================================================================
raw = "AAABBBAA"
grouped = [(k, list(g)) for k, g in it.groupby(raw)]
print("\n[groupby 相邻]", grouped)
# 说明：A 出现在两头 → 被当成两组！需要全局分组就要先排序：
g2 = [(k, len(list(g))) for k, g in it.groupby(sorted(raw))]
print("[groupby 先 sort]", g2)      # 同键归并

# =====================================================================
# 四、tee：把一个迭代器克隆成可独立消费的多个副本
# =====================================================================
src = (i * 2 for i in range(5))
a, b = it.tee(src, 2)                # 生两个独立可遍历的“视图”
print("\n[tee] a:", list(a))
print("[tee] b(同值可再遍历):", list(b))

# =====================================================================
# 五、accumulate：前缀和 / 累积
# =====================================================================
print("\n[accumulate 累加]", list(it.accumulate([1, 2, 3, 4])))
print("[accumulate *乘用法]",
      list(it.accumulate([1, 2, 3, 4], lambda x, y: x * y)))  # 前缀积1,2,6,24

# =====================================================================
# 六、无限/循环迭代器：count / cycle / repeat，务必与 islice/takewhile 截断
# =====================================================================
print("\n[count 从5步进2] 前5项:", list(it.islice(it.count(5, 2), 5)))
cyc = it.cycle("AB")
print("[cycle 无限循环] 前6项:", "".join(it.islice(cyc, 6)))
print("[repeat 重复] 前4项:", list(it.islice(it.repeat(1), 4)))

# =====================================================================
# 七、组合数学：product / permutations / combinations
# =====================================================================
print("\n[product 笛卡尔积]", list(it.product([1, 2], [3, 4])))
print("[permutations(有序) 2取]", list(it.permutations("AB", 2)))
print("[combinations(无序) 2取]", list(it.combinations("ABC", 2)))

# 提示：类别结果巨大时不是 list()，而是 for 流式取需要的前 N 个
head = list(it.islice(it.permutations(range(10), 5), 3))
print("[组合结果流式取前3]", head)

# =====================================================================
# 八、常见“速查”写法（官方 Recipes 风格）
# =====================================================================
def grouper(iterable, n):
    """把元素按 n 个一组打包（最后一组可能不完整）。"""
    # itertools.zip_longest 按最长补齐，fillvalue 填 None
    args = [iter(iterable)] * n
    return zip_longest_fixed(args)

from itertools import zip_longest
def zip_longest_fixed(args):
    yield from zip_longest(*args, fillvalue=None)

print("\n[分组工具示例] grouper(3):",
      [list(g) for g in grouper([1, 2, 3, 4, 5], 3)])

# =====================================================================
# 小结
# =====================================================================
# itertools 全是惰性迭代器 → 高效、省内存；要用到下标切片用 islice、
# 全局去重分组先 sort、无穷序列必须截断、组合类别别 list 全量。
