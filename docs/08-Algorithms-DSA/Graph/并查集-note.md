---
title: "并查集（Union-Find，路径压缩与按秩合并）"
tags: [数据结构, 图论, 算法, 并查集]
date: 2026-08-30
---

# 并查集（Union-Find，路径压缩与按秩合并）

## 定义

并查集（Union-Find，也叫 Disjoint Set Union / DSU）是一种维护**不相交集合（disjoint sets）**的数据结构，只支持两类操作：**find(x)** —— 查询元素 x 属于哪个集合（返回该集合的「代表元」，通常即树根）；**union(x, y)** —— 把 x、y 各自所在的集合合并成一个。它要解决的是**动态等价关系维护**问题：初始时每个元素自成一个集合，之后不断交替执行「合并」与「查询两个元素是否同属一集」，每次操作只关心元素归属哪个集合，完全不关心集合内部的元素排列。

它适用于「元素全集固定、只合并不拆分、只增不减」的等价类划分问题，典型场景包括：无向图中判断两点是否连通、统计连通分量（省份/朋友圈）个数、无向图判环、Kruskal 最小生成树中的环检测、以及各类离线倒序的动态连通性问题。

核心特征是：底层用若干棵「父指针树」表达集合，树根作为整个集合的唯一身份标识；配合**路径压缩（path compression）**与**按秩合并（union by rank）**两个启发式后，单次操作摊还复杂度降至 $O(\alpha(n))$（$\alpha$ 为反阿克曼函数，工程上可视作常数），这也是它成为图论与竞赛中最常用数据结构之一的原因。

注意它的边界：不支持把一个大集合拆回小集合（除非用撤销/持久化变体）、不关心集合内部元素、节点必须编号化（非连续键需改用字典实现）——这些是它与普通「连通分量统计」「哈希表」等方案的重要区分点。

## 原理

思路：用数组 `parent` 存每个节点的父指针，`parent[i] = i` 表示 i 是树根；用根作为集合代表元。`find` 沿父指针向上找根即得集合身份；`union` 把一棵树的根接到另一棵树的根下。若不加优化，连续合并可能造出一条链，`find` 最坏退化为 $O(n)$——因此必须引入两个启发式（本笔记主题）：

**路径压缩**：在 `find(x)` 的过程中，把沿途访问到的每个节点直接挂到根下（`parent[·] = 根`）。这样后续对这些节点的查询只需一步。递归写法为 `parent[x] = find(parent[x])`，迭代写法分两趟：先找根，再沿原路把节点全部指向根（或采用「路径减半」只跳一半）。其本质是「查询即重构」：把本次查询的开销预支为未来若干次查询的加速，是摊还分析（amortized analysis）的典型例子。

**按秩合并**：为每棵树维护秩 rank（树高的上界）。合并时**总是把秩较小的根接到秩较大的根上**，仅当两树秩相等时，被指向一方的秩才 +1。这一规则保证任意树高 ≤ $\lfloor \log_2 n \rfloor$：秩为 0 的树至少有 $2^0=1$ 个节点，两棵秩 r 的树合并成秩 r+1 时，节点数至少翻倍为 $2^{r+1}$，即

$$
2^{\text{rank}(t)} \le \text{size}(t) \implies \text{height}(t) \le \log_2 n .
$$

于是单次 find（未压缩时）最坏也是 $O(\log n)$，从根本上阻止链式退化。

两个启发式合用的复杂度为：任意 $m$ 次操作（含 $n$ 次初始化）总代价 $O(m\,\alpha(n))$，其中 $\alpha$ 是**反阿克曼函数**（增长极慢，$\alpha(n) \le 4$ 对所有现实规模的 n 成立），因此实践中单次 find/union 可视为常数级。推导直觉：路径压缩把 find 的代价「记账」到被拍平/抬高的边上，而按秩合并限定了树高的增长，两者共同使总跳数保持在近线性内（严格证明见 Tarjan & van Leeuwen 1984）。注意缺失其一效果即打折：只用按秩合并，单次最坏 $O(\log n)$；只用路径压缩，摊还 $O(\log n)$——必须**两者都做**才拿到 $O(\alpha(n))$。

标准流程（下标按 0/1 统一即可）：`make-set(i): parent[i]=i, rank[i]=0` → `find(x): 若 parent[x]≠x 则递归压缩; 返回 parent[x]` → `union(x,y): rx=find(x), ry=find(y); 若 rx==ry 返回 False（已同集/成环）; 秩小接秩大; 秩相等则目标秩+1; 返回 True`。

## 应用

典型使用场景：① 无向图连通性查询与**连通分量计数**（省份数量、朋友圈、冗余连接，n 个点每合并成功一次分量数 -1）；② **无向图判环**——Kruskal 依次考察边时，若两端点 find 结果相同说明加入即成环，应跳过；③ **离线动态连通性**——「只删边、只提问」的问题先处理完所有删除、再倒序把边加回来用并查集回答，是经典离线技巧；④ 网格/矩阵连通块（如被水包围的岛屿、感染扩散，把相邻同属性格 union 起来）；⑤ 等价关系类问题（账号合并、同义句判定、区间等价）。快速上手步骤：1) 按数据规模建 `parent/rank` 数组，确认节点编号从 0 还是 1 开始；2) 实现 find（大数据量优先迭代版）+ 路径压缩；3) 实现 union：先 find 判同集，再按秩合并；4) 遍历输入边依次 union，并在合并成功处更新答案统计量（分量数、是否成环等）。

边界条件与常见坑（❌/✅）：
- ❌ `union` 里不先 `find` 就直接 `parent[y] = x`：父子关系错乱，可能产生环、丢失代表元。
- ❌ 两元素已在同一集合还执行合并/秩更新逻辑：应提前 `return False`（该返回值正好用于判环）。
- ❌ 合并成功后忘记把「连通分量个数」等统计量 -1，或忘记判断 union 返回值就盲目计数。
- ❌ 只做路径压缩不做按秩合并，或反之：退化为摊还 $O(\log n)$ 甚至更差，违背设计初衷。
- ❌ 大规模数据（$n \ge 10^5$）用递归 find：未压缩的深树或 Python 默认递归上限会触发 RecursionError，工程代码建议写迭代两趟式 find。
- ❌ 节点编号从 1 开始却申请了长度 n 的数组（应为 n+1）；邻接矩阵只扫上三角避免重复 union。
- ✅ 判根统一用 `parent[x] == x`，切勿用 `rank[x] == 0` 之类判断（根被接走后其 rank 不变，会误判）。
- ✅ 只想统计连通块数时，可用 `count` 初始 n、每次 union 成功自减的写法，避免每次全量扫描。

```python
class UnionFind:
    """并查集：路径压缩 + 按秩合并。单次操作摊还 O(α(n))，可视为常数。"""

    def __init__(self, n: int):
        self.parent = list(range(n))  # parent[i] 为 i 的父指针；parent[i]==i 表示 i 是根
        self.rank = [0] * n           # rank[i] 为树高上界（秩）
        self.count = n                # 当前不相交集合（连通分量）个数

    def find(self, x: int) -> int:
        """迭代版 find + 完整路径压缩：返回 x 所在集合的根，并把沿途节点全部挂到根下。"""
        root = x
        while self.parent[root] != root:   # 第一趟：向上找根
            root = self.parent[root]
        while self.parent[x] != x:         # 第二趟：沿原路压缩，每个节点直接指向根
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, x: int, y: int) -> bool:
        """合并 x、y 所在集合；返回是否真的发生了合并（False 表示二者已同集/会成环）。"""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:                       # 已在同一集合：直接返回，不加边（可用于判环）
            return False
        if self.rank[rx] < self.rank[ry]:  # 按秩合并：秩小的根接到秩大的根上
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]: # 仅当两树秩相等时，新根的秩 +1
            self.rank[rx] += 1
        self.count -= 1                    # 合并成功 → 集合数减一
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


# ---- 案例：LeetCode 547「省份数量」----
# 输入 n×n 邻接矩阵 is_connected：is_connected[i][j]==1 表示城市 i、j 直接相连，
# 直接或间接相连的城市属于同一个省份，求省份总数。
def find_circle_num(is_connected) -> int:
    n = len(is_connected)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):          # 矩阵对称，只扫上三角，避免重复合并
            if is_connected[i][j]:
                uf.union(i, j)             # 有边 → 把两个城市并入同一集合
    return uf.count                        # 剩余集合数即省份数


if __name__ == "__main__":
    g1 = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]  # 0-1 相连，2 孤立
    print(find_circle_num(g1))              # 输出 2
    g2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 三个城市各自孤立
    print(find_circle_num(g2))              # 输出 3
```

**案例详解**：以 `g1` 为例，初始化后 parent=[0,1,2]、rank 全 0、count=3。扫描到 `is_connected[0][1]==1` 执行 `union(0,1)`：`find(0)=0`、`find(1)=1`，二者秩相等（0==0）且 `rank[0] < rank[1]` 不成立，故把 1 挂到 0 下：`parent[1]=0`，两树秩相等故 `rank[0]` 由 0 升为 1，count 3→2。上三角其余位置全为 0，不再合并，最终 `count=2`，即 {0,1} 与 {2} 两个省份。`g2` 中没有任何 `union` 成功，count 保持 3，输出 3。注意路径压缩在代码中的体现：若之后查询 `find(1)`，第一趟沿 parent[1]=0 找到根 0，第二趟发现 parent[1] 已等于根、无需改动；若树更深（如 0←1←2 的链），查询 find(2) 会把 1、2 一并直接指向 0，下次所有查询都是 O(1) 跳转——这正是压缩带来的摊还收益。复杂度小结：初始化 O(n) 时间/空间；对 n×n 邻接矩阵共执行约 n²/2 次 union，每次摊还 O(α(n))，故整体 $O(n^2\,\alpha(n))$；省去路径压缩或按秩合并其一则升为 $O(n^2\log n)$ 量级。

---
## 关联
- 前置：[[BFS与DFS-note]]（先理解「连通分量 / 可达性」概念；BFS/DFS 是静态图上的遍历式解法，并查集是其支持动态加边的等价工具）
- 类似：[[哈希表-note]]（区别是哈希表解决「键→值」的映射存取、不做集合合并；并查集维护「元素→所属集合代表」且集合会动态合并，parent 也可用字典实现以支持非连续键）
- 进阶：[[快速排序-note]]（Kruskal 最小生成树 = 按边权排序 + 并查集判环合并，总体复杂度由排序与并查集共同决定）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：并查集（路径压缩 + 按秩合并） | 父指针森林、根为代表元；压缩拍平路径、按秩限制树高，摊还 $O(\alpha(n))$ | 频繁「动态加边 + 连通查询/判环/分量计数」、Kruskal 判环、离线倒序动态连通 |
| BFS/DFS 染色遍历 | 从每个未访问点出发遍历整块连通分量并染色标记 | 图已固定（静态）时统计连通块/可达性；无法廉价支持动态加边，加边后要重跑或重建 |
| 链表式并查集（加权合并） | 每个集合维护一条链表，union 时把小链表整体并入大链表 | 需要经常枚举集合内全部元素（find 摊还 $O(\log n)$，比本文慢，但枚举元素代价低） |
| 朴素合并（无启发式） | 直接 parent[y]=x 挂接，不压缩不按秩 | 仅一次性小规模使用（n 很小）；频繁操作会退化成 O(n) 链，不推荐 |

---
## 参考
- [Wikipedia: Disjoint-set data structure](https://en.wikipedia.org/wiki/Disjoint-set_data_structure)
- [CP-Algorithms: Disjoint Set Union（含路径压缩/按秩合并与复杂度说明）](https://cp-algorithms.com/data_structures/disjoint_set_union.html)
- [Tarjan & van Leeuwen: Worst-case analysis of set union algorithms, JACM 31(2), 1984](https://doi.org/10.1145/62.2160)
- [CLRS《Introduction to Algorithms》第 4 版（MIT Press），Data Structures for Disjoint Sets 章节](https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/)

---
## 具体案例
- [[并查集（Union-Find，路径压缩与按秩合并） 实战示例]](并查集_sample.py)
