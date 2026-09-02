---
title: "哈希表与 Python dict 原理"
tags: [数据结构, 哈希表, Python, 查找]
date: 2026-08-30
---

# 哈希表与 Python dict 原理

## 定义

哈希表（Hash Table，又称散列表）是一种以“键—值”为存储单位的抽象数据结构：它借助哈希函数（hash function）把键映射为数组下标，从而把“按键查找”问题转化为“按下标直接访问”，平均 $O(1)$ 完成查找、插入、删除。

它要解决的核心问题是：在一个**无序、动态增长、键类型任意**的集合上，如何只凭键本身快速定位其对应值——顺序扫描要 $O(n)$，有序数组二分查找要 $O(\log n)$，而哈希表在平均意义下只要 $O(1)$。

核心特征有三：一是**哈希函数**把无限键空间压缩到有限槽位（典型如除法散列 $h(k)=k \bmod m$）；二是**冲突解决策略**（开放寻址或链地址法）处理多个键映射到同一槽的情况；三是**装载因子**（load factor，$\alpha = n/m$）控制表密度并及时扩容，防止性能退化。

它是“用空间换时间”的典型：为了让大多数键各自落在不同槽位，表容量通常大于元素个数，换来常数级访问。适用范畴几乎覆盖所有“等值命中”型查询：集合去重、频次统计、键值配置、缓存记忆化、数据库哈希索引、编译器符号表、图遍历的 visited 记录等；Python 的 dict / set、Java 的 HashMap / HashSet 都是它的工业实现。

局限同样明显：内部无序（Python 3.7+ 仅保证**插入顺序**，仍无下标/切片/范围查询），只支持等值命中，键必须可哈希（hashable），且内存开销明显大于同规模数组。

## 原理

**思路**：数组天生支持按整数下标 $O(1)$ 访问，可惜键不是整数下标——哈希表做的事就是把“任意键”折算成“下标”。整体结构 = 一段连续数组（槽）+ 哈希函数 + 冲突处理 + 动态扩容，核心公式：

$$h(k)=k \bmod m \qquad\text{（除法散列，} m \text{ 为槽数）}$$

Python 中任意对象先调用内建 `hash(k)` 得到整数哈希，再对表长取模（或取掩码）得到起点。整数哈希在 CPython 中等于该整数对 $2^{61}-1$ 取模，所以小整数满足 `hash(n) == n`，唯一特例 `hash(-1) == -2`（-1 曾是内部哨兵值）；字符串哈希基于带进程级随机盐的 SipHash（可用环境变量 `PYTHONHASHSEED` 固定盐用于调试复现），其目的正是让攻击者难以构造“大量同哈希”的键来制造碰撞型拒绝服务。

冲突为什么不可避免：键空间远大于槽数，鸽巢原理保证必然有不同键映射到同一槽。两大解决流派：

- **链地址法**：每个槽挂一条链表，冲突键串在同一条链上。装载因子 $\alpha=n/m$，随机键假设下成功查找平均比较约 $1+\alpha/2$ 次、失败约 $1+\alpha$ 次。推导：某键所在链的期望长度为 $1+(n-1)/m \approx 1+\alpha$，成功查找只需比较链中排在自己前面的键，约一半，故 $1+\alpha/2$；把 $\alpha$ 钳制为常数即得平均 $O(1)$。
- **开放寻址**（CPython dict 采用）：冲突时沿探测序列找下一个空槽。CPython 使用带扰动的伪随机探测 $i=(5i+\text{perturb}+1) \bmod m$：$i$ 每轮乘 5 加 1 避免小步长聚集，`perturb`（原始哈希每轮右移 5 位）负责把序列在大范围内“打散”，兼顾局部探测与全局覆盖。删除时不能直接清空槽，必须留下“墓碑/哑元”（dummy）标记，否则会切断后续键的探测链；这些哑元在扩容时被统一清除。理论对照：线性探测失败查找的期望探测次数约 $\frac{1}{2}\left(1+\frac{1}{(1-\alpha)^2}\right)$，随 $\alpha \to 1$ 超线性爆炸——这正是必须把装载因子控制在约 $2/3$ 以下的原因。

CPython dict 工程细节：3.6 起改为“紧凑布局”（compact dict），键值条目按插入顺序追加进连续数组，另有一张索引表 `dk_indices` 把哈希探测结果指向条目下标，因此既保插入顺序又省内存（3.7 起该顺序成为语言规范）；PEP 412 的 split-table 让同类实例的 `__dict__` 共享同一张键表、仅值数组各自独立，进一步省内存。当装载因子逼近约 $2/3$ 触发扩容：新容量约翻倍，所有键重新散列（rehash），单次扩容 $O(n)$，但均摊到每次插入仍是 $O(1)$。

复杂度汇总与推导：

| 操作 | 最好 | 平均 | 最坏 |
|------|------|------|------|
| 查找 / 插入 / 删除 | $O(1)$ | $O(1)$（$\alpha$ 受控为常数） | $O(n)$（键全部冲突） |

插入的 $O(1)$ 是**摊还**意义：扩容使容量几何增长，$n$ 次插入累计的 rehash 代价不超过 $O(n+m_0)$，均摊到每次为常数；最坏 $O(n)$ 出现在所有键哈希到同一槽（如攻击构造的碰撞序列）。空间复杂度 $O(n)$，但 dict 每键要额外存一份哈希、索引与条目，常系数和内存占用明显大于同规模 list。

## 应用

典型场景：
- **去重与存在性判断**：`x in set`、图遍历的 visited 集合、爬虫 URL 去重。
- **频次统计**：词频、字符计数、日志聚合，直接 `Counter` 或 `defaultdict(int)`。
- **缓存与记忆化**：递归 memoization、`functools.lru_cache`、把“状态→结果”装进 dict 的自顶向下动态规划。
- **一趟配对/反查**：两数之和、前缀和配对、用 dict 做“值→下标”反查，把 $O(n^2)$ 暴力降到 $O(n)$。
- **键值存储与索引**：配置项、路由参数解析、数据库哈希索引的核心思想。

快速上手步骤：
1. 选好键：只用**可哈希且不可变**的对象（str / int / tuple / frozenset）；多字段复合键拼成 tuple。
2. 选对结构：只判断存在用 set；键值映射用 dict；需默认值用 `defaultdict`；纯计数用 `Counter`。
3. 少写“先查再写”：用 `d.get(k, 默认)`、`d.setdefault(k, 默认)`、`d.pop(k, 默认)` 一次完成读写，避免两次查找。
4. 迭代与排序：3.7+ 遍历即插入顺序；要排序输出用 `sorted(d.items(), key=...)`。
5. 权衡内存：若键本来就是连续小整数，直接用 list 当下标映射更快更省，无需 dict。

边界条件 / 易错点（❌ 坑 / ✅ 正确做法）：
- ❌ 可变对象当键（list、dict、set）→ `TypeError: unhashable type`；换成 tuple / frozenset。
- ❌ 自定义类覆写 `__eq__` 后，Python 3 会自动把 `__hash__` 置为 None，对象变成不可哈希；需同时显式定义 `__hash__`，并保证 `a == b ⇒ hash(a) == hash(b)`（哈希不变式）。
- ❌ 遍历 dict 的同时增删元素 → `RuntimeError: dictionary changed size during iteration`；先 `list(d)` 做快照再删，或用字典推导重建。
- ✅ 判断键是否存在用 `k in d`；不要用 `d.get(k) is None` 判断——键存在但值为 None/0/False/"" 时会误判为“不存在”。
- ✅ 记住 dict 顺序语义分界：3.6 前无序，3.7 起语言规范保证“插入顺序”；但它仍无下标/切片/范围查询。
- ✅ 慎用 float 当键：`0.0` 与 `-0.0` 哈希相同视为同键；`float('nan')` 与自身不相等，存入后无法按原键取回。
- ✅ 最坏 $O(n)$ 不是纸面威胁：对**外部输入可控的键**（自定义对象、特定整数序列）可构造碰撞拖垮性能；CPython 已用随机盐 + SipHash 防护字符串，自己实现哈希表时也要用随机化哈希。
- ✅ 手写开放寻址表必须处理“墓碑”删除标记与扩容 rehash；生产环境直接使用内建 dict / set，非教学场景不要重复造轮子。

```python
"""
哈希表核心原理演示：手写链地址法哈希表 + Python dict 实战。
用法：直接运行本文件（python hash_table_demo.py）。
"""
from collections import defaultdict, Counter


class HashTable:
    """最小可用链地址法哈希表：槽内用 list 当桶（演示原理，非性能最优）。"""

    def __init__(self, capacity=8):
        self.capacity = capacity                 # 槽数 m
        self.buckets = [[] for _ in range(capacity)]
        self.size = 0                            # 已存键值对数 n
        self.threshold = 0.7                     # 装载因子阈值

    def _hash(self, key):
        # 散列函数：内建 hash(key) 先算出整数哈希，再对槽数取模得到下标
        return hash(key) % self.capacity

    def _resize(self):
        """装载因子超阈值：容量翻倍后把所有键值重新散列（rehash）。"""
        old = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        for bucket in old:                       # 旧桶里的每个键必须重算下标
            for k, v in bucket:
                self.buckets[self._hash(k)].append((k, v))

    def __setitem__(self, key, value):
        idx = self._hash(key)
        bucket = self.buckets[idx]
        for i, (k, _) in enumerate(bucket):      # 冲突：先在桶内线性找同键
            if k == key:                         # 命中 → 覆盖旧值
                bucket[i] = (key, value)
                return
        bucket.append((key, value))              # 未命中 → 追加到桶尾
        self.size += 1
        if self.size / self.capacity > self.threshold:
            self._resize()                       # 摊还 O(1) 的关键步骤

    def __getitem__(self, key):
        for k, v in self.buckets[self._hash(key)]:
            if k == key:
                return v
        raise KeyError(key)                      # 桶内找不到 → 键不存在

    def __contains__(self, key):
        return any(k == key for k, _ in self.buckets[self._hash(key)])

    def __len__(self):
        return self.size


def two_sum(nums, target):
    """两数之和：dict 记录「值 -> 下标」，一趟扫描把 O(n^2) 暴力降到 O(n)。"""
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:                   # 互补值出现过 → 找到答案
            return [seen[target - x], i]
        seen[x] = i                              # 否则登记当前值及其下标
    return []


if __name__ == "__main__":
    # 1) 手写链地址法哈希表：插入、覆盖、存在性判断
    ht = HashTable()
    for k, v in [("apple", 3), ("banana", 5), ("cherry", 8), ("apple", 9)]:
        ht[k] = v
    print("ht['apple'] =", ht["apple"])          # 9：同键后写覆盖先写
    print("'cherry' in ht =", "cherry" in ht)    # True

    # 2) 触发扩容：插入 10 个键使 size 超过 0.7*8≈5，验证 rehash 后数据完好
    for i in range(10):
        ht[f"num{i}"] = i
    print("len(ht) =", len(ht))                  # 13（3 + 10）
    print("'num7' in ht =", "num7" in ht)        # True：扩容重散列后仍可命中

    # 3) Python dict：3.7+ 保持插入顺序；用 get 一趟完成计数
    d = {}
    for w in "banana apple cherry apple".split():
        d[w] = d.get(w, 0) + 1                   # 一次 get 兼顾“查”与默认值
    print("dict 计数:", d)                        # 顺序即单词首次出现顺序

    # 4) 更专业的计数与默认值写法
    print("Counter:", Counter("banana apple cherry apple".split()))
    dd = defaultdict(int)
    for w in "banana apple".split():
        dd[w] += 1
    print("defaultdict:", dict(dd))

    # 5) 边界条件演示：不可哈希对象不能当键
    try:
        d[[1, 2]] = "oops"                       # list 可变 → 不可哈希
    except TypeError as e:
        print("TypeError:", e)

    # 6) 哈希表经典应用：两数之和
    print("two_sum([2, 7, 11, 15], 9) =", two_sum([2, 7, 11, 15], 9))  # [0, 1]
```

**案例详解**：运行后输出 `ht['apple'] = 9`（同键后写覆盖）与 `'cherry' in ht = True`；插入 `num0`~`num9` 触发两次扩容（size 分别超过 $0.7\times8$ 与 $0.7\times16$），随后 `len(ht) = 13` 且 `'num7' in ht = True`，证明翻倍扩容 + 全量 rehash 后所有键仍然可查——容量几何增长保证扩容次数只有 $O(\log n)$ 次，总 rehash 代价 $O(n)$ 摊还到每次插入即 $O(1)$。`dict 计数` 按“首次出现顺序”打印 `banana → apple → cherry`，直接验证 Python 3.7+ 的插入顺序保证；`Counter`、`defaultdict` 是等价的更专业写法。`d[[1, 2]]` 触发 `TypeError: unhashable type: 'list'`——可变对象哈希值不可信，故不可哈希。`two_sum` 返回 `[0, 1]`：每步只查“互补值是否出现过”，一趟 $O(n)$ 完成，体现“用哈希空间换时间”的典型收益。手写版刻意省略删除操作：开放寻址型删除必须留“墓碑”标记，否则会截断探测链、让后续键失联，这正是生产级实现的核心难点，日常开发直接使用内建 dict / set 即可。

---
## 关联
- 类似：[[二分查找-note]]（区别是二分查找建立在“有序数组 + 比较”之上，$O(\log n)$，支持有序遍历与范围查询；哈希表建立在“散列 + 数组槽”之上，平均 $O(1)$ 但无序、只支持等值命中，且键必须可哈希）
- 进阶：[[堆排序与优先队列-note]]（哈希擅长“等值定位”、堆擅长“动态取最值”，二者常组合使用：用 dict 记录元素在堆中的下标，即可把“修改堆内任意元素”降到 $O(\log n)$，这是 Dijkstra 等算法配合优先队列时的关键优化手法）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：哈希表 / Python dict | 哈希函数把键映射为数组下标，开放寻址解决冲突，装载因子约 2/3 触发翻倍扩容 | 无序动态键值映射，频繁等值查找/插入/删除，去重、计数、缓存 |
| 链地址法哈希表（HashMap 风格） | 槽内挂链表，冲突键串链，链过长时（如 Java 8+ 阈值 8）转红黑树 | 需要容忍高冲突率、元素极多，或语言默认实现即链地址的场景 |
| 有序数组 + 二分查找 | 先排序，再用比较折半收缩搜索区间 | 数据基本静态、需要范围查询/有序输出，可接受 $O(\log n)$ |
| 平衡二叉搜索树（红黑树等） | 靠元素比较维持动态有序，天然支持前驱后继 | 既要动态插入删除，又要有序遍历/范围统计，且不能接受哈希的无序性 |

---
## 参考
- [Python 官方文档：Mapping Types — dict](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict)
- [CPython 源码：Objects/dictobject.c](https://github.com/python/cpython/blob/main/Objects/dictobject.c)
- [PEP 412 — Key-Sharing Dictionary](https://peps.python.org/pep-0412/)
- [Python 数据模型：object.__hash__（可哈希对象约束）](https://docs.python.org/3/reference/datamodel.html#object.__hash__)
- [Wikipedia：Hash table（复杂度与冲突策略综述）](https://en.wikipedia.org/wiki/Hash_table)

---
## 具体案例
- [[哈希表与 Python dict 原理 实战示例]](哈希表与 Python dict 原理_sample.py)
