---
title: "堆排序与优先队列（Heap）"
tags: [堆排序, 优先队列, 二叉堆, 排序算法]
date: 2026-08-30
---

# 堆排序与优先队列（Heap）

## 定义

堆（Heap，下文特指二叉堆 Binary Heap）是一种基于**完全二叉树**的数据结构：除最后一层外每层都被填满，最后一层节点从左到右连续排列。正因为形状固定，堆可以直接用**一维数组**存储、用下标演算父子关系，不需要指针，天然缓存友好。

堆的核心约束是**堆序性质（heap property）**：最大堆要求任意父节点 ≥ 其两个子节点（堆顶即全局最大值），最小堆要求任意父节点 ≤ 子节点（堆顶即全局最小值）。注意它只保证“部分有序”——兄弟节点之间、左右子树之间没有任何大小约束，这是堆比有序数组/平衡树“更廉价”的根本原因。

优先队列（Priority Queue）是一个抽象数据类型（ADT）：支持 O(1) 查看、O(log n) 取出当前优先级最高（或最低）的元素、O(log n) 插入任意新元素。二叉堆是该 ADT 最经典的实现方式，二者经常被混用，但严格说前者是“接口”、后者是“数据结构”。

堆排序（Heapsort）是把最大堆当作“自动吐出最大值的黑盒”、反复把堆顶交换到数组末尾的**原地、不稳定**比较排序。它最吸引人的性质是**最坏情况也能保证 O(n log n)**，且不需要额外 O(n) 空间——这正是它相对快速排序（最坏 O(n²)）与归并排序（需 O(n) 辅助空间）的差异化价值。

适用范畴：需要**动态维护极值 / TopK / 第 K 大（小）**的流式场景；“每次取当前最优”驱动的贪心与图算法（Dijkstra 最短路、Prim 最小生成树、哈夫曼编码）；以及无法接受排序退化、又要原地排序的场合。

## 原理

**思路**：优先队列要回答的问题是“剩下未处理的元素里谁最大/最小”。朴素做法每次线性扫描是 O(n)。堆的做法是维护一棵满足堆序的完全二叉树，让任何增删只沿**一条根到叶的路径**传导——路径长度等于树高 O(log n)——于是插入与取出都只要 O(log n)。堆排序则更进一步：不申请辅助数组，直接把待排序数组建成最大堆，再通过 n-1 次“堆顶 ↔ 堆尾交换 + 缩小堆”完成升序。

为什么数组就能表示树：完全二叉树可以按层序紧凑装入数组，第 i 个节点的孩子与父亲可纯下标计算（0 基）：

$$left(i)=2i+1,\qquad right(i)=2i+2,\qquad parent(i)=\left\lfloor\frac{i-1}{2}\right\rfloor$$

两个基本原语：**上浮 sift-up**（插入时用）——新元素追加到数组末尾，若比父节点更“优”就与父交换并继续向上，直到满足堆序；**下沉 sift-down**（取出堆顶、建堆时用）——把节点与更“优”的孩子交换并一路向下，直到不比任何子节点差。

堆排序三步（升序用最大堆）：① 建堆：自底向上从最后一个非叶节点 `n//2 - 1` 逐个下沉；② 把堆顶（当前最大值）与堆的最后一个元素交换，最大值即落入最终位置；③ 堆长度减一，对新堆顶执行一次下沉恢复堆序，重复 n-1 次。循环不变量是“已交换到末尾的后缀已排好序且互不干扰”。

复杂度推导：建堆时高度为 h 的节点至多下沉 h 次，而高度为 h 的节点约有 n/2^{h+1} 个：

$$T_{\text{build}}=\sum_{h=0}^{\lfloor\log_2 n\rfloor}\Big\lceil\frac{n}{2^{h+1}}\Big\rceil\cdot h \le n\sum_{h=1}^{\infty}\frac{h}{2^h}=O(n)$$

随后 n-1 次“交换 + 下沉”各耗 O(log n)，故总复杂度 O(n) + O(n log n) = **O(n log n)**（最好/平均/最坏均为此量级，全等键等极端输入因比较提前终止会更早结束，但上界不变）。空间上完全原地、无递归栈，**额外空间 O(1)**；代价是跳跃式数组访问缓存不友好、常数明显大于快速排序，且不稳定。优先队列在堆上的对应：插入 = 末尾追加 + 上浮 O(log n)，取出堆顶 = 堆顶与末尾交换 + 下沉 O(log n)，窥视堆顶 O(1)，`heapify` 整体建堆 O(n)。

## 应用

典型使用场景：① TopK / 第 K 大（小）：维护大小为 K 的堆，总代价 O(n log K)；② 合并 K 个有序序列（多路归并）每次取 K 路中最小者，总 O(n log K)；③ 数据流中位数：大根堆 + 小根堆各存一半并保持大小差 ≤ 1；④ 定时器/任务调度：到期时间最早的先执行；⑤ Dijkstra / Prim / 哈夫曼编码等“每次取当前最小”的算法。

快速上手（Python）：① 标准库 `heapq` 本身就是最小堆，`heappush / heappop / heapify / heapreplace` 直接可用；② 需要最大堆就存相反数 `-x`，弹出时再取负还原；③ 一次性把列表建成堆必须用 `heapq.heapify`（O(n)），逐个 `heappush` 是 O(n log n)，量级不同；④ 求最大的 K 个元素应维护**大小为 K 的最小堆**（堆顶是“门槛”），求最小的 K 个元素反之维护大小为 K 的最大堆。

易错点与边界条件：

- ❌ 把 `heapq` 当最大堆用——它是最小堆，`heappop` 永远先给最小值；要最大堆必须“存负 + 取负还原”。
- ❌ 认为建堆是 O(n log n)：只有自顶向下逐个插入才是 O(n log n)；Floyd 自底向上下沉是 O(n)。能一次到位的列表不要循环 push。
- ❌ 下标算错：0 基数组里父节点是 `(i-1)//2`、最后一个非叶节点是 `n//2-1`；把 1 基公式 `2i / 2i+1` 直接套用会漏节点或越界。
- ❌ 下沉时不做越界检查：必须先判 `left < n` 再取左孩子，比较右孩子前还要确认 `right < n`。
- ❌ 对空堆执行 `heappop` 会抛 `IndexError`；`heapreplace` 要求堆非空且语义是“弹出堆顶再压入”，只应在新元素确实更优时调用。
- ❌ 堆排序不稳定：相等元素可能互换相对顺序；业务要求稳定时改用归并排序，或给元素附加次序编号。
- ❌ 元素入堆后再修改其取值，堆序不会自动修复（堆没有通用的 decrease-key 操作）：要么重建，要么用“懒删除 + 新值重新入堆”绕过（Dijkstra 的标准做法）。
- ✅ 查看堆顶（peek）是 O(1)，直接读 `h[0]`，不要“pop 再 push”。
- ✅ 第 K 大用大小为 K 的最小堆；当 K 接近 n 时全排序反而更快，堆方案只在 K ≪ n 或数据流式到达时占优。

```python
# -*- coding: utf-8 -*-
"""堆排序 + 优先队列（二叉堆）示例：原地堆排序、heapq 最小堆、TopK 技巧。"""
import heapq


def sift_down(arr, n, i):
    """最大堆下沉：把 arr[i] 一路换到合适位置，使 [0, n) 区间保持大顶堆性质。"""
    while True:
        largest = i
        left, right = 2 * i + 1, 2 * i + 2      # 0 基下标：左孩子 2i+1，右孩子 2i+2
        if left < n and arr[left] > arr[largest]:
            largest = left                      # 左孩子更大则记录之
        if right < n and arr[right] > arr[largest]:
            largest = right                     # 右孩子存在才比较，防越界
        if largest == i:                        # 两个孩子都不比它大 -> 已到位
            break
        arr[i], arr[largest] = arr[largest], arr[i]  # 与较大的孩子交换
        i = largest                             # 继续下沉被换下来的原节点


def build_max_heap(arr):
    """Floyd 自底向上建堆：从最后一个非叶节点 n//2-1 开始逐个下沉，总代价 O(n)。"""
    for i in range(len(arr) // 2 - 1, -1, -1):
        sift_down(arr, len(arr), i)


def heap_sort(arr):
    """原地升序堆排序：建最大堆 -> 堆顶(最大值)换到末尾 -> 对新堆顶下沉。"""
    build_max_heap(arr)
    for end in range(len(arr) - 1, 0, -1):      # end: 未排序区间的末尾下标
        arr[0], arr[end] = arr[end], arr[0]     # 当前最大值落入最终位置
        sift_down(arr, end, 0)                  # 堆长缩短为 end，新堆顶重新下沉


def top_k_largest(nums, k):
    """返回 nums 中最大的 k 个数：维护大小为 k 的最小堆，堆顶是第 k 大的门槛。"""
    if k <= 0:
        return []
    h = []
    for x in nums:
        if len(h) < k:
            heapq.heappush(h, x)                # 堆未满，直接入堆
        elif x > h[0]:
            heapq.heapreplace(h, x)             # 比门槛大才替换(pop+push 的原子版)
    return sorted(h, reverse=True)              # 仅便于展示，不是算法的一部分


if __name__ == "__main__":
    a = [4, 10, 3, 5, 1, 8, 7, 6]
    heap_sort(a)
    print("heap_sort:", a)                      # [1, 3, 4, 5, 6, 7, 8, 10]

    nums = [4, 10, 3, 5, 1, 8, 7, 6, 9]
    print("top_k   :", top_k_largest(nums, 3))  # [10, 9, 8]

    pq = []                                     # heapq 最小堆 = 最常用的优先队列
    for v in (3, 1, 2):
        heapq.heappush(pq, v)
    print("min     :", heapq.heappop(pq))       # 1：永远先弹出最小值

    maxq = [-x for x in (3, 1, 2)]              # 要最大堆：元素取负后入堆
    heapq.heapify(maxq)
    print("max     :", -heapq.heappop(maxq))    # 3：弹出后记得取负还原
```

**案例详解**：`sift_down` 是整套算法唯一的核心子程序，比较前先判 `left/right < n` 杜绝越界，无需交换即 break 退出；`build_max_heap` 从最后一个非叶节点（8 个元素时为下标 3）自底向上下沉，结束后数组形态为 `[10, 6, 8, 5, 1, 3, 7, 4]`——堆顶 10 是最大值，其余位置只是“部分有序”，这正是堆与有序数组的本质区别。`heap_sort` 随后每轮把堆顶交换到 `end`（最大值落入最终位置）、把堆长缩短为 `end` 再对新的堆顶下沉，如此往复最终输出严格升序 `[1, 3, 4, 5, 6, 7, 8, 10]`，全程原地、无递归。`top_k_largest` 演示“小堆守大门槛”技巧：堆里始终只有 K 个元素，空间 O(K)、单元素 O(log K)，远优于全排序后再截断。最后两段验证 `heapq` 的语义：它是最小堆，故 pop 得 1；要最大堆必须存 `-x`，弹出后再取负还原得 3——忘记还原符号是最高频的笔误。

---
## 关联
- 前置：[[快速排序-note]]（同为原地、不稳定的比较排序；对照“快排靠划分、堆排靠堆化”，理解两者如何用不同的机制换最坏情况保证）
- 类似：[[归并排序-note]]（区别：归并稳定、需 O(n) 辅助空间，适合链表与外部排序；堆排序原地 O(1) 空间但不稳定；二者最坏都达到 O(n log n)）
- 进阶：[[BFS与DFS-note]]（Dijkstra 最短路可视为“优先队列驱动的搜索”，其底层正是二叉堆；先读懂堆，再回头看带权图算法会豁然开朗）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案（二叉堆 / 堆排序） | 完全二叉树 + 数组存储；上浮/下沉维护堆序；排序 = 建堆 + 反复“换顶 + 下沉” | 动态插入且反复取极值、TopK / 第 K 大、流式数据、需要原地且最坏 O(n log n) 的排序 |
| 快速排序（分治划分） | 选基准把序列划分成“左小右大”两段再递归，靠划分结果定位元素 | 通用排序首选（C++ std::sort、Java 原生类型排序的 introsort 都以快排为核心），平均常数最小；怕最坏退化或要求稳定时不选 |
| 归并排序（分治合并） | 先拆到单元素，再两两合并出有序段，稳定且适合顺序访问 | 链表排序、外部排序、求逆序对、需要稳定性（Python sorted/Timsort 即稳定归并家族）；代价是 O(n) 辅助空间 |
| 平衡二叉搜索树（sortedcontainers 等） | 维护元素全序的平衡树，天然有序、支持查找/删除任意元素 | 除取极值外还要删除任意元素、求前驱后继或整体有序遍历；实现与常数开销都明显高于堆 |

---
## 参考
- [heapq — 堆队列算法（Python 官方文档）](https://docs.python.org/3/library/heapq.html)
- [Heapsort - Wikipedia](https://en.wikipedia.org/wiki/Heapsort)
- [Binary heap - Wikipedia](https://en.wikipedia.org/wiki/Binary_heap)
- [Priority queue - Wikipedia](https://en.wikipedia.org/wiki/Priority_queue)
- [Introduction to Algorithms, 4th Edition - MIT Press（CLRS，第 6 章 Heapsort 与优先队列）](https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/)

---
## 具体案例
- [[堆排序与优先队列（Heap） 实战示例]](堆排序与优先队列_sample.py)
