---
title: "图的遍历：BFS 与 DFS"
tags: [算法, 图论, 搜索]
date: 2026-08-30
---

# 图的遍历：BFS 与 DFS

## 定义

图的遍历（Graph Traversal）指从某个（或某些）起点出发，按某种规则访问图中**所有可达顶点**、且**每个顶点恰好访问一次**的过程。它是图论算法的基础操作：连通性判断、最短路径、环检测、拓扑排序、连通分量、二分图判定等，几乎都以遍历为骨架。

解决的核心问题有两个：一是**不重**——需要一个 visited 标记避免重复访问（否则有环图会无限循环）；二是**不漏**——需要保证所有从起点可达的顶点都被访问（若图不连通，还要在外层循环扫描所有顶点，否则孤立分量会被漏掉）。

两大主流方案的区别只在于**下一轮探索哪个顶点**：BFS（Breadth-First Search，广度优先搜索）用一个队列按"先进先出"逐层扩散，先访问完距起点第 $k$ 层的全部顶点，再访问第 $k+1$ 层；DFS（Depth-First Search，深度优先搜索）用栈（递归的系统栈或显式栈）按"后进先出"沿一条路径一路走到底，撞墙后回溯，再走下一条支路。

适用范畴：只要问题可建模成"从起点出发在图/树/网格/状态空间上探索"，就适用。BFS 偏重"找最近"（无权最短路、最少步数），DFS 偏重"穷举与回溯"（连通性、路径枚举、状态搜索）。

## 原理

**思路**：遍历的正确性由两件事保证——visited 标记保证"不重"，从而必然终止；对图结构的系统性扫描保证"不漏"。BFS 与 DFS 的唯一差别是把"待访问顶点"放进队列还是栈，因此代码骨架几乎相同，只是数据结构不同。

**BFS 机制**：起点入队并标记；每次从队首取出顶点 $v$ 访问，再把 $v$ 所有**未标记**的邻居入队并立即标记（入队即标记，而不是出队才标记——否则同一顶点会被多个邻居重复入队）。队列 FIFO 性质保证了出队顺序按"到起点的距离"非降排列，因此**无权图中顶点首次被访问时所处的层数就是最短距离**，数学上可用归纳证明：距离为 $d$ 的顶点一定在第 $d$ 轮扩展中、且不晚于距离为 $d+1$ 的顶点被访问。

**DFS 机制**：访问当前顶点后，取它的第一个未访问邻居深入；该分支穷尽后**回溯**到上一个顶点继续取下一个邻居。递归版本天然表达回溯（系统调用栈）；迭代版本用显式栈模拟。把"访问顶点的时间点"分别定义为前序/后序，可支撑不同任务：前序做拓扑排序的反向（后序反转），三色标记法（白/灰/黑）检测有向环。

**复杂度推导**：每个顶点至多入队/入栈一次，贡献 $O(1)$，共 $O(V)$；每条边在扫描邻居时被检查一次（无向图每条边被两个端点各检查一次，仍是常数倍），共 $O(E)$。故邻接表下最好/最坏/平均均为：

$$
T(V,E) = O(V+E)
$$

若用邻接矩阵存图，扫描一个顶点的全部邻居要扫一整行 $O(V)$，总复杂度退化为 $O(V^2)$。空间上，BFS 队列最坏容纳整个最宽的一层（如星形图）为 $O(V)$，平均约等于"层宽"；DFS 递归/栈深度等于当前路径长度，链状图最坏 $O(V)$，平均等于"树高"，通常比 BFS 省内存。两者在网格 $R\times C$ 上则都是 $O(RC)$ 时间、$O(RC)$ 空间（visited 数组）。

## 应用

典型使用场景：

- **无权图最短路 / 最少步数**（BFS）：社交"几度人脉"、迷宫/推箱子最少步数、单词接龙，层数即答案。
- **连通性 & 连通分量**（BFS/DFS 皆可）：判断两点是否连通、统计孤立岛屿数量。
- **环检测、拓扑排序、强连通分量**（DFS）：如课程安排可行性（检测依赖环）。
- **二分图判定**（BFS 染色）：相邻层染相反颜色，冲突即非二分图。
- **穷举搜索 / 回溯**（DFS）：八皇后、数独、排列组合、表达式树遍历，配合剪枝。

快速上手步骤：① 选定存储方式（邻接表/邻接矩阵/网格），② 准备 visited 标记（集合或布尔数组），③ 把起点放入数据结构（BFS 用 `deque` 队列，DFS 用递归或栈），④ 循环取顶点→访问→扩展未标记邻居，⑤ 若图可能不连通，外层再遍历所有顶点作为起点。

❌/✅ 易错点清单：

- ❌ visited 在**出队时**才标记 → 同一顶点被多个邻居重复入队，队列膨胀、甚至超时。✅ 应在**入队/入栈的同时**标记。
- ❌ 直接写 `list.pop(0)` 当队列 → 每次 O(n) 搬移，BFS 退化到 $O(V^2)$。✅ 用 `collections.deque` 的 `popleft()`，O(1)。
- ❌ 只从单个起点遍历不连通图 → 孤立分量被漏掉，统计岛屿/连通分量出错。✅ 外层 `for` 扫描所有未访问顶点。
- ❌ 忘判 visited 就开始 DFS 递归 → 有环图死循环直到 RecursionError。✅ 入口先标记，或迭代版用显式栈。
- ❌ 深度极大的图（如 $10^5$ 长链）用递归 DFS → 系统栈溢出（Python 默认限制约 1000 层）。✅ 改迭代栈，或 `sys.setrecursionlimit`（治标不治本）。
- ❌ 无向图只存一条边 → 邻接表两个方向都要加，否则连通性判断出错。
- ❌ BFS 求最短路时误用于**带权图** → 层数不再等于代价，需换 Dijkstra（BFS 是它的无权特例）。
- ❌ 网格 BFS 越界访问 → 每次扩展邻居前先检查 `0 <= r < rows` 且 `0 <= c < cols`。

```python
# -*- coding: utf-8 -*-
"""图的遍历：BFS 与 DFS —— 可运行示例（Python 3.8+）"""
from collections import deque


def bfs_order(graph, start):
    """BFS 广度优先遍历：队列先进先出，天然按"层"扩散。"""
    visited = {start}          # 入队即标记，杜绝重复入队
    order = []                 # 记录访问顺序
    q = deque([start])
    while q:
        v = q.popleft()        # 队首出队，O(1)
        order.append(v)
        for nxt in graph[v]:           # 扩展 v 的全部邻居
            if nxt not in visited:
                visited.add(nxt)
                q.append(nxt)          # 未访问过的邻居入队
    return order


def bfs_dist(graph, start):
    """BFS 求无权图单源最短距离：首次被访问时的层数即最短距离。"""
    dist = {start: 0}
    q = deque([start])
    while q:
        v = q.popleft()
        for nxt in graph[v]:
            if nxt not in dist:        # 首次到达 = 最短（层数递增性质）
                dist[nxt] = dist[v] + 1
                q.append(nxt)
    return dist


def dfs_recursive(graph, v, visited, order):
    """DFS（递归版）：先访问，再沿邻居一头扎到底，分支穷尽后回溯。"""
    visited.add(v)             # 进入时标记，防止重复访问
    order.append(v)
    for nxt in graph[v]:
        if nxt not in visited:
            dfs_recursive(graph, nxt, visited, order)
    return order


def dfs_iterative(graph, start):
    """DFS（迭代版）：用显式栈模拟系统递归栈，避免深图递归爆栈。"""
    visited = {start}
    order = []
    stack = [start]
    while stack:
        v = stack.pop()        # 后进先出：沿刚发现的分支深入
        order.append(v)
        for nxt in graph[v]:
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return order


def bfs_grid_min_steps(grid, sr, sc):
    """网格（迷宫）BFS：0 是路、1 是墙，返回从 (sr, sc) 到每格的最少步数。"""
    rows, cols = len(grid), len(grid[0])
    dist = [[-1] * cols for _ in range(rows)]   # -1 表示尚未到达（含墙）
    dist[sr][sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # 下、上、右、左
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols \
                    and grid[nr][nc] == 0 and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
    return dist


if __name__ == "__main__":
    # 无向图：A-B-C-D-A 围成环，B 再连出 E（邻接表存储）
    g = {
        "A": ["B", "C"],
        "B": ["A", "C", "D", "E"],
        "C": ["A", "B", "D"],
        "D": ["B", "C"],
        "E": ["B"],
    }
    print("BFS 顺序：", bfs_order(g, "A"))
    print("BFS 距离：", bfs_dist(g, "A"))
    print("DFS 递归：", dfs_recursive(g, "A", set(), []))
    print("DFS 迭代：", dfs_iterative(g, "A"))

    # 迷宫 3 行 5 列，左上角 (0, 0) 出发
    maze = [
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
    ]
    print("迷宫最少步数表：", bfs_grid_min_steps(maze, 0, 0))
```

**案例详解**

示例图是"四边形环 + E 挂在 B 上"的无向图。`bfs_order` 从 A 出发：第 1 层访问 B、C，第 2 层经 B 才首次发现 D、E（C 的邻居都已访问），因此顺序恒为 `['A','B','C','D','E']`，且 `bfs_dist` 给出的 `{'A':0,'B':1,'C':1,'D':2,'E':2}` 正好等于各自所在层数——这就是 BFS 求无权最短路"首次到达即最短"的直观体现。DFS 递归版沿 A→B→C→D 深入到底再回溯补上 E，顺序同样是 `['A','B','C','D','E']` 但**含义不同**（这是深度优先的前序，不是按层）；迭代版因压栈顺序产生 `['A','C','D','B','E']`，同样是合法的 DFS 前序——遍历顺序本身不是唯一答案，只要"不重不漏"即可。`bfs_grid_min_steps` 把四方向扩散的层数写入 dist 表，最终输出中 `(0,0)` 为 0、(1,1) 为 2、(0,4) 和 (2,4) 为 6，墙为 -1，证明 BFS 在网格上同样按"环状波前"推进，给出各格最少步数。

---
## 关联
- 前置：[[哈希表-note]]（visited 判重基于哈希集合的 $O(1)$ 查找，是两种遍历高效的前提）
- 类似：[[并查集-note]]（都解决连通性问题；区别：并查集只回答"是否连通"，BFS/DFS 还给出遍历顺序、距离与路径）
- 进阶：[[堆排序与优先队列-note]]（把 BFS 的普通队列换成按权值排序的优先队列，即 Dijkstra 最短路的骨架）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案 1：BFS（队列） | 先进先出逐层扩散，首次到达即无权最短路 | 无权图最短路径、最少步数、按层遍历、连通分量、二分图判定 |
| 本文方案 2：DFS（递归/显式栈） | 后进先出一路到底、回溯穷举，空间常更省 | 连通性、环检测、拓扑排序、强连通分量、回溯搜索（八皇后/数独） |
| 替代方案 1：迭代加深 DFS（IDDFS） | 逐步放宽深度上限反复 DFS，兼得 BFS 最优性与 DFS 空间 | 解很浅但图极深/无限状态空间，空间受限又要最短解 |
| 替代方案 2：Dijkstra | BFS 队列换优先队列，按累计权值出队 | 带权图单源最短路；A* 加启发式后用于大规模地图寻路 |

---
## 参考
- [Breadth-first search - Wikipedia](https://en.wikipedia.org/wiki/Breadth-first_search)
- [Depth-first search - Wikipedia](https://en.wikipedia.org/wiki/Depth-first_search)
- [Python 标准库文档：collections.deque](https://docs.python.org/3/library/collections.html#collections.deque)
- [《算法导论》（CLRS）第 4 版（MIT Press）](https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/)

---
## 具体案例
- [[图的遍历：BFS 与 DFS 实战示例]](图的遍历：BFS 与 DFS_sample.py)
