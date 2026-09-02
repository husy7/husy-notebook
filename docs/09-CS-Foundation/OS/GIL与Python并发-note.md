---
title: "Python GIL 与并发影响"
tags: [Python, GIL, 并发, CPython]
date: 2026-08-30
---

# Python GIL 与并发影响

## 定义

GIL（Global Interpreter Lock，全局解释器锁）是 CPython（Python 官方解释器）在实现层面引入的一把互斥锁：它保证**同一时刻整个进程内只有一个线程在执行 Python 字节码**，其余线程只能等待。它要解决的核心问题是：CPython 的内存管理基于引用计数（reference counting），而引用计数的增减本身不是线程安全的——两个线程若同时对同一对象执行 `Py_INCREF`/`Py_DECREF`，会产生竞态，导致对象被提前释放（悬空指针）或永不被释放（内存泄漏）。与其为每个对象、每个容器都做细粒度加锁，设计者选择了一把覆盖整个解释器全局状态的"大锁"，换来实现简单、单线程性能好、难以出现死锁。

核心特征：锁的粒度极粗（全解释器一把）、进入/释放快、对单线程程序几乎零感知。代价也很明确——多线程执行纯 Python 计算时被强制串行化，**无法利用多核 CPU 的并行能力**。适用范围：GIL 属于 CPython 的实现细节而非 Python 语言规范，官方 `threading`/`concurrent.futures` 文档并不保证跨实现的多线程并行语义；CPython 与 PyPy 有 GIL，Jython、IronPython 等不依赖该内存模型的实现没有 GIL，Python 3.13 起实验性的 free-threaded（`--disable-gil` 构建，即 3.13t）也已移除它。

一句话概括影响：**IO 密集型任务用多线程可以获得并发，CPU 密集型任务用多线程只能获得串行**——前者要归功于阻塞期间 GIL 的主动释放，后者则必须转向多进程、协程或能主动释放 GIL 的 C 扩展。

## 原理

**为什么这样设计（动机）**：CPython 用引用计数管理对象生命周期，还维护全局对象状态（如 interned strings、内存分配器）。若多个线程真正并行执行，这些共享结构都需要各自的锁或原子操作，复杂度高且会拖慢单线程；在 GIL 诞生的年代（多核尚未普及），选择一把全局锁是最省事且对单线程最快的方案。1990 年代末曾出现过移除 GIL 的实验分支，结果单线程性能下降明显、且几乎全部 C 扩展需要重写，方案被否决——这也是 GIL 长期保留的历史原因。

**核心机制**：持锁线程在"执行若干字节码"与"检查是否需要让出"之间循环，主要释放路径有三条：

1. **时间片到期**：Python 3.2 起从"每 N 条字节码"改为基于时间——线程持锁运行满 `sys.getswitchinterval()`（默认 5ms）后设置 drop request 并释放 GIL，通知等待线程竞争；
2. **阻塞调用主动释放**：线程执行 `sleep`、文件/网络 IO、等待锁或条件变量等阻塞操作时，C 层会先释放 GIL 再进入内核等待（对应 `Py_BEGIN_ALLOW_THREADS`/`Py_END_ALLOW_THREADS` 宏），IO 完成后重新获取——这正是 IO 密集多线程能并发的原因；
3. **C 扩展显式释放**：第三方扩展可在长计算中用相同宏主动放锁，让其它 Python 线程插空运行（numpy 的多数 ufunc、numba 的 `nogil=True` 即如此）。

**原子性粒度**：在 GIL 下，单条字节码的执行是原子的，纯 C 实现且期间不放锁的内置操作（如一次 `list.append`、单次 `dict` 读写）也近似原子；但 `counter += 1` 会被编译为 LOAD/ADD/STORE 三条字节码，两条之间可能发生线程切换，因此"读-改-写"复合逻辑仍然必须用 `threading.Lock` 等显式同步。可以用一个流程图描述持锁线程的轮转：

```mermaid
flowchart LR
    A["线程A：持有GIL，逐条执行字节码"] -->|"运行满 switchinterval(默认5ms)"| B["线程A：置 drop request，释放GIL"]
    B --> C{"GIL 被谁抢到？"}
    C -->|线程B 抢到| D["线程B：持有GIL，执行字节码"]
    D -->|"遇到 sleep/IO/等待锁等阻塞调用，主动释放"| C
    D -->|"运行满时间片，释放"| B
    C -->|线程A 再次抢到| A
```

**为什么 CPU 密集多线程不加速**：由于纯 Python 计算期间 GIL 从不释放，"可并行比例"近似为 0，多线程收益受 Amdahl 定律约束。若单核串行耗时 $T_1$、可并行执行的比例为 $p$、核数为 $N$，则：

$$
S(N)=\frac{T_1}{T_N}=\frac{1}{(1-p)+\frac{p}{N}},\qquad N\ \text{核时的加速比}
$$

GIL 场景下对字节码执行有 $p \approx 0$，故 $S(N) \approx 1$——4 个线程跑 CPU 任务约等于串行执行 4 份任务（加上切换开销甚至更慢）。反之 IO 密集任务在等待时 $p \to 1$，多线程便能重叠等待、获得接近线性的并发收益。

**当前演进**：PEP 703（2023 年接受）提出把 GIL 变为可选——Python 3.13 提供实验性 free-threaded 构建（`--disable-gil`，版本号带 `t` 后缀），3.14 继续完善；它靠更细粒度的 per-object 锁与原子操作保证线程安全，代价是单线程性能略有回退，且依赖 C API 全局状态的扩展（如旧版 numpy）需要适配后才能安全运行。

## 应用

**典型场景判断（写并发代码前的第一步）**：先用 profiling 确认任务是 CPU 密集（大量纯 Python 循环/计算）还是 IO 密集（网络请求、读写文件、sleep）。随后按"IO 用线程或协程、CPU 用进程或 C 扩展"的口诀选择并发模型。

**快速上手步骤**：

1. 用 `sys.getswitchinterval()` 查看当前 GIL 切换间隔（默认 `0.005` 秒），了解你所在解释器的调度粒度；
2. IO 密集任务：直接使用 `concurrent.futures.ThreadPoolExecutor` 或 `asyncio`，线程数可远超 CPU 核数；
3. CPU 密集任务：改用 `ProcessPoolExecutor`（每个进程持有独立解释器与独立 GIL），或把热点下沉到会释放 GIL 的库（numpy、numba `nogil=True`）；
4. 线程间需要共享可变状态时，用 `threading.Lock`/`RLock` 保护临界区，并借助 `with` 语句保证异常时也能释放锁；
5. 多进程间共享数据走 `multiprocessing.Manager`、`Queue` 或共享内存，避免大对象整体复制。

**常见坑**：① 误以为"有 GIL 就线程安全"——GIL 只保证字节码级原子，复合操作照样有竞态；② CPU 密集任务盲目开大量线程，不仅不加速反而因切换开销变慢；③ 把线程模型一棍子打死——只要底层调用会释放 GIL（IO、部分 C 库运算），多线程仍可并行；④ `ProcessPoolExecutor` 要求任务函数可 pickle，Windows 下用 spawn 启动，代码须包在 `if __name__ == "__main__"` 保护中；⑤ 随意调小 `sys.setswitchinterval()` 只会增加切换频率、放大竞态概率，并不能提升吞吐。

```python
# gil_demo.py —— 用四组实验观察 GIL：CPU/IO 密集对比、字节码竞态、加锁修复
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def cpu_task(_i, n=2_000_000):
    """CPU 密集：纯 Python 循环，执行字节码期间持有 GIL、不会主动释放。"""
    s = 0
    for j in range(n):
        s += j * j % 997
    return s


def io_task(_i):
    """IO 密集：time.sleep 在 C 层实现，会主动释放 GIL，让出解释器。"""
    for _ in range(20):
        time.sleep(0.01)
    return 0


def bench(fn, workers, executor_cls):
    """fn 为模块级函数（可 pickle），线程池/进程池均可运行。"""
    with executor_cls(max_workers=workers) as pool:
        start = time.perf_counter()
        list(pool.map(fn, range(workers)))   # 等待全部任务结束
        return time.perf_counter() - start


if __name__ == "__main__":                     # Windows spawn 模式必需
    # 实验 1：CPU 密集 —— 多线程被 GIL 串行化，多进程才能吃满多核
    print(f"CPU密集  4线程: {bench(cpu_task, 4, ThreadPoolExecutor):.2f}s")
    print(f"CPU密集  4进程: {bench(cpu_task, 4, ProcessPoolExecutor):.2f}s")

    # 实验 2：IO 密集 —— 阻塞时释放 GIL，线程与进程差距很小
    print(f"IO密集   4线程: {bench(io_task, 4, ThreadPoolExecutor):.2f}s")
    print(f"IO密集   4进程: {bench(io_task, 4, ProcessPoolExecutor):.2f}s")

    # 实验 3：把切换间隔调到 0.1ms，让"读-改-写"竞态稳定复现
    sys.setswitchinterval(0.0001)              # 默认 0.005s(5ms)
    N, WORKERS, counter = 500_000, 8, 0

    def add_race():
        global counter
        for _ in range(N):
            counter += 1                       # LOAD/ADD/STORE 之间可能被切走

    threads = [threading.Thread(target=add_race) for _ in range(WORKERS)]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"无锁计数: {counter}（期望 {WORKERS * N}，通常偏小）")

    # 实验 4：threading.Lock 保证临界区互斥，结果恢复正确
    lock, counter2 = threading.Lock(), 0

    def add_locked():
        global counter2
        for _ in range(N):
            with lock:                         # with 确保异常时也释放锁
                counter2 += 1

    threads = [threading.Thread(target=add_locked) for _ in range(WORKERS)]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"加锁计数: {counter2}（期望 {WORKERS * N}）")
```

**案例详解**：实验 1 中，4 个线程共享一个 GIL，4 份 CPU 任务基本串行执行，耗时约等于单任务 × 4；而 4 个进程各有独立 GIL，可在 4 核上并行，耗时约降到 1/4（输出形如 `CPU密集 4线程: 0.92s / 4进程: 0.25s`，随机器核数与主频浮动，此为示意值）。实验 2 中 `sleep` 期间线程主动放锁，4 线程与 4 进程耗时接近（都约 `0.21s`），说明 IO 场景线程已足够且省去进程开销。实验 3 把切换间隔从 5ms 调小到 0.1ms 后，`counter += 1` 的三条字节码之间被频繁打断，多个线程读到旧值再写回，出现丢失更新（lost update），结果稳定小于期望值 4,000,000。实验 4 用 `with lock` 把"读-改-写"包成临界区后计数精确等于期望值——这正是 GIL 之外仍需显式同步的实证：GIL 管"解释器"，锁管"临界区"。

---
## 关联
- 前置：[[进程线程与协程-note]]
- 类似：[[内存管理-note]]（区别是内存管理笔记关注对象生命周期、引用计数与分配器本身，GIL 则是对这些内部状态的互斥保护机制）
- 进阶：[[IO模型-note]]
- Python 侧：[[python上下文管理-note]]（`with` 管理 Lock/RLock 的获取与释放）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：CPython 多线程 + GIL | 解释器级互斥锁串行化字节码执行，阻塞时主动释放以保证 IO 并发 | IO 密集且需要在线程间共享可变状态的场景 |
| multiprocessing 多进程 | 每个进程一个解释器、一把独立 GIL，进程级并行 | 纯 Python 实现的 CPU 密集计算，需吃满多核 |
| asyncio 协程 | 单线程事件循环协作式调度，无锁竞争、开销最小 | 海量网络连接的高并发 IO（C10K 类） |
| free-threaded CPython（3.13t/--disable-gil） | 移除全局锁，以 per-object 锁与原子操作保证线程安全 | 依赖库已兼容、希望多线程直接并行计算的实验场景 |

---
## 参考
- [Python 官方术语表：global interpreter lock](https://docs.python.org/3/glossary.html#term-global-interpreter-lock)
- [Python 官方文档：threading —— 基于线程的并行](https://docs.python.org/3/library/threading.html)
- [PEP 703 – Making the Global Interpreter Lock Optional in CPython](https://peps.python.org/pep-0703/)
- [David Beazley：Understanding the Python GIL（PyCon 2010 演讲）](https://www.youtube.com/watch?v=Obt-vMVdM8s)

---
## 具体案例
- [[Python GIL 与并发影响 实战示例]](Python GIL 与并发影响_sample.py)
