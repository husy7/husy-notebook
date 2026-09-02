---
title: "迭代器 iterators"
tags: [Python, 迭代器, 迭代协议, __iter__, __next__]
date: 2026-08-29
---

# 迭代器 iterators

本篇专攻**迭代器**这一细分点（父级主题见《生成器和迭代器》）：迭代协议是什么、可迭代对象与迭代器的区别、如何自定义一个迭代器类。

## 定义

迭代器（iterator）是**实现了迭代协议的对象**：在 Python 中，一个对象只要实现 `__iter__()` 与 `__next__()` 两个方法，就能被 `for` 逐个拿值。`for x in obj` 的底层动作就是把 `obj` 转成迭代器并反复 `next`，直到遇到 `StopIteration` 为止。

| 概念 | 实现 | 能否被 `for` |
|------|------|:---:|
| **可迭代 iterable** | 只需 `__iter__()`（返回迭代器） | ✅（for 会调 iter(obj)） |
| **迭代器 iterator** | 需 `__iter__()` **和** `__next__()` | ✅ |

- **它解决什么问题**：把"数据源（集合、文件、流）"与"逐个取值"解耦——无论底层是内存列表还是逐行读的大文件，消费方都只用统一的 `next()` 接口拿下一个值，无需关心值如何产生、何时产生。
- **核心特征**：惰性（lazy）——值在每次 `next()` 时才产出，而非一次性全部装入内存；一次性（single-pass）——迭代器遍历完即耗尽，不能回头重来。
- **与可迭代对象的关系**：可迭代对象只需 `__iter__`，可被反复 `for`；迭代器同时有 `__next__`，是一次性数据流。`list`/`tuple`/`str`/`dict`/`set` 是可迭代对象；`iter(lst)` 的结果、file 对象、生成器是迭代器。
- **适用范畴**：`for` 循环、序列解包、`in` 成员判断、内建惰性工具（`zip`/`map`/`enumerate`/`filter`）的底层取值、自定义容器类接入 `for`、大文件与无限序列的流式处理。
- **判定口诀**：有 `__next__` 的是迭代器（一次流）；只有 `__iter__` 的是可迭代对象（可反复遍历）。

## 原理

- **迭代协议的两方法分工**：`__iter__()` 负责返回一个迭代器——对容器类通常新建/返回一个迭代器对象，对迭代器本身通常 `return self`，从而让"可迭代"成立；`__next__()` 负责每次产出一个值，并在没有更多值时抛出 `StopIteration`，这是协议规定的**终止信号**。
- **for 的等价展开（为什么必须抛 StopIteration）**：`for` 实际执行 `it = iter(obj); while True: try: x = next(it) except StopIteration: break`。`__next__` 若在尽头忘记抛 `StopIteration`，`for` 就永远等不到终止信号而无限循环；反之，只要协议齐全，任何对象都能被 `for` 消费。
- **为什么分"可迭代"与"迭代器"两类（很多对象既是又是的 trick）**：容器数据可反复遍历（每次 `for` 都新建迭代器），而遍历过程本身一次性——两职责分离使 `list` 能重复 `for`，`iter(list)` 却只能走一遍。`range`、生成器、文件对象是"既是又是"：自身可迭代，且自身就是迭代器（一次性）。
- **序列协议兜底机制**：对象没有 `__iter__` 时，`iter(obj)` 退而尝试"下标协议"——从下标 0、1、2… 依次调 `obj[0]`、`obj[1]`…，直到 `IndexError` 为止。这让只实现 `__getitem__` 的旧式对象也能被迭代。
- **惰性求值的代价**：迭代器不保存已产出的值，因此不支持下标与 `len`；需要下标/长度时必须先 `list(it)` 物化，而物化会**耗尽**该迭代器（这是内存换一次性，需重新 `iter()` 才能再遍历）。

## 应用

- **典型使用场景**：① `for` 遍历集合/文件；② 大文件逐行读取，惰性省内存；③ 自定义类接入 `for`（树的遍历器、倒计时、状态机式数据流）；④ 与 `zip`/`map`/`enumerate`/`filter` 组合做惰性变换；⑤ 手动 `next()` 精确控制取值节奏（如只取前 N 个即停，配合 `itertools.islice`）。
- **快速上手步骤**：写一个类 → 实现 `__iter__`（通常 `return self`）→ 实现 `__next__`（每次返回一个值，耗尽前必须 `raise StopIteration`）→ 直接交给 `for` 或 `next()` 使用。
- **内建工具速查**：

| 函数/位置 | 作用 |
|-----------|------|
| `iter(obj)` | 拿到 obj 的迭代器；无 `__iter__` 时按"下标 0,1,2…"的序列协议兜底 |
| `next(it, default)` | 取下一个；给了 default 时到头返回 default 而不抛异常 |
| `zip` / `map` / `enumerate` / `filter` | 都返回**迭代器**，惰性产出 |

- **常见坑（易错点）**：
  - ❌ 混淆"可迭代"与"迭代器"：`for` 能遍历的未必是一次性迭代器；`list` 可重复 `for`，迭代器不行。✅ 记住：迭代器 = 有 `__next__` 的一次流。
  - ❌ 在 `__next__` 里忘记抛 `StopIteration` → `for` 永不结束。✅ 在耗尽处 `raise StopIteration`。
  - ❌ 需要下标/长度时对迭代器误用 `it[0]` / `len(it)`（迭代器不支持）。✅ 先 `list(it)`，但注意它会耗尽迭代器。
  - ❌ 字典默认迭代遍历的是**键**。✅ 需 `values()`/`items()`；dict 自身顺序在 3.7+ 为插入序。
  - ⚠️ 文件对象本身是迭代器，`for` 一遍后指针已到底，需重新打开或 `seek` 才能再读。

```python
# ===== 示例 1：从可迭代对象取迭代器，手动 next =====
lst = [1, 2, 3]
it = iter(lst)          # list 只有 __iter__；iter() 调它拿到迭代器
print(next(it))         # 1 —— 每次 next 前进一个值
print(next(it))         # 2
# lst 之后还能被再次 for（容器可反复遍历）；it 却已走到 2，是一次性流

# ===== 示例 2：for 的底层机制（等价展开） =====
it = iter(lst)          # 1. 先拿到迭代器
while True:
    try:
        x = next(it)    # 2. 反复 next 取值
    except StopIteration:   # 3. 迭代器耗尽抛 StopIteration -> for 正常退出
        break
    print(x)

# ===== 示例 3：自定义迭代器类 Countdown =====
class Countdown:
    """倒计时迭代器：自己既是可迭代对象也是迭代器"""
    def __init__(self, n):
        self.n = n

    def __iter__(self):     # 迭代器协议：返回迭代器，此处返回自身
        return self

    def __next__(self):     # 每次 next 产出一个值
        if self.n <= 0:     # 耗尽边界：必须抛 StopIteration，否则 for 永不结束
            raise StopIteration
        self.n -= 1
        return self.n

for x in Countdown(3):
    print(x)                # 2 1 0

# ===== 案例详解 =====
# for Countdown(3) 时：iter() 调 __iter__ 拿到迭代器（即自身）；
# 随后 for 反复 next()：第一次 __next__ 使 n=2 并返回 2，第二次返回 1，
# 第三次返回 0；第四次 n=0 不再 >0，抛 StopIteration，for 捕获后正常结束。
# 结果依次打印 2、1、0。若删掉 __next__ 里的 raise StopIteration，
# 循环将无限进行（每次只返回 0），这正是"忘抛 StopIteration"的经典坑。
# 若再实现 __getitem__(self, i) 而非 __iter__，iter(obj) 也能按下标协议兜底，
# 但那是旧式序列风格，自定义迭代器推荐直接实现 __iter__ + __next__。
```

---
## 关联
- 前置：[[for 循环]]（for 的底层就是"转迭代器 + 反复 next + 捕获 StopIteration"）
- 类似：[[生成器 generator]]（区别是____生成器靠 `yield` 自动实现迭代协议、状态由函数栈帧保存，无需手写 `__iter__`/`__next__`；手写迭代器类则须显式维护状态并在耗尽时自行抛 `StopIteration`）
- 类似：[[可迭代对象 iterable]]（区别是____可迭代对象只需 `__iter__`，可反复遍历；迭代器额外需要 `__next__`，是一次性数据流，二者是"容器 vs 遍历过程"的关系）
- 进阶：[[collections.abc.Iterator 与 Iterable]]（用抽象基类做 isinstance 判定与子类注册）、[[itertools]]（迭代工具组合，详见另篇）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：手写迭代器类（`__iter__` + `__next__`） | 用类字段显式保存遍历状态，每次 `next` 产一个值，耗尽抛 `StopIteration`，取值逻辑完全可控 | 需要以类封装复杂遍历逻辑（树/图遍历、倒计时、状态机式数据流）或复用同一套状态 |
| 替代方案：生成器 `yield` | 把迭代器状态交给函数栈帧，`yield` 处挂起/恢复，自动生成迭代器，代码最简 | 快速惰性序列、流水线式数据处理、无限序列（配合 itertools） |
| 替代方案：内建惰性工具 `iter`/`next`/`zip`/`map`/`enumerate`/`filter` | 不自实现协议，直接组合现成迭代器做惰性变换 | 常规遍历与变换、只取前几个元素、大文件/大列表的内存友好处理 |

---
## 参考
- [Python 官方文档：迭代器类型（Iterator Types）](https://docs.python.org/3/library/stdtypes.html#typeiter)
- [Python 教程：迭代器章节（Iterators）](https://docs.python.org/3/tutorial/classes.html#iterators)

---
## 具体案例
- [[迭代器 实战示例]](迭代器_sample.py)
