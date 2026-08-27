# -*- coding: utf-8 -*-
"""
python生成器与迭代器_sample.py
知识点：Python 生成器（Generator）与迭代器（Iterator）
运行方式：python python生成器与迭代器_sample.py
"""

# ============================================================
# 1. 类实现迭代器：手动实现 __iter__ + __next__（迭代协议）
# ============================================================
class Countdown:
    """倒计时迭代器：手动管理状态与终止条件"""

    def __init__(self, start):
        self.n = start

    def __iter__(self):
        return self  # 迭代器必须返回自身

    def __next__(self):
        if self.n <= 0:
            raise StopIteration  # 协议规定的终止信号
        self.n -= 1
        return self.n + 1


# ============================================================
# 2. 生成器函数：等价逻辑，用 yield 自动保存状态
# ============================================================
def countdown_gen(start):
    """与 Countdown 类等价的生成器函数"""
    n = start
    while n > 0:
        yield n  # 每次在此暂停，n 的状态被自动保留
        n -= 1


# ============================================================
# 3. 生成器表达式：惰性版推导式（注意是圆括号）
# ============================================================
def gen_expr_demo():
    squares = (x * x for x in range(10**8))  # 瞬间创建，不占内存
    print("生成器表达式前两个值:", next(squares), next(squares))  # 0 1


# ============================================================
# 4. 数据管道：生成器链式处理（惰性流水线）
# ============================================================
def read_lines(nums):
    """模拟逐行读取数据源"""
    for i in nums:
        yield i


def only_even(it):
    """只保留偶数"""
    for x in it:
        if x % 2 == 0:
            yield x


# ============================================================
# 5. 常见坑演示：迭代器是一次性的（耗尽后静默返回空）
# ============================================================
def one_shot_demo():
    gen = countdown_gen(3)
    print("第一次遍历:", list(gen))  # [3, 2, 1]
    print("第二次遍历:", list(gen))  # [] —— 已耗尽，不报错但为空


# ============================================================
# 6. 需要多次遍历：用 itertools.tee 分裂迭代器
# ============================================================
def tee_demo():
    from itertools import tee
    gen2, gen3 = tee(countdown_gen(3))
    print("tee 分裂后:", list(gen2), list(gen3))  # [3, 2, 1] [3, 2, 1]


# ============================================================
# 7. 无限序列：生成器天然支持惰性求值
# ============================================================
def fib():
    """斐波那契无限序列"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def infinite_demo():
    from itertools import islice
    print("斐波那契前10项:", list(islice(fib(), 10)))


# ============================================================
# 8. 生成器的高级特性：send / return 值 / yield from
# ============================================================
def echo():
    """用 send() 向生成器内部传值"""
    while True:
        received = yield
        if received is None:
            break
        print("生成器收到:", received)


def advanced_demo():
    # --- send() ---
    g = echo()
    next(g)          # 启动生成器，执行到第一个 yield
    g.send("hello")  # 输出: 生成器收到: hello
    g.send("world")  # 输出: 生成器收到: world
    g.close()

    # --- 生成器中 return 的值进入 StopIteration.value ---
    def gen_with_return():
        yield 1
        return "done"

    g2 = gen_with_return()
    print("yield 值:", next(g2))  # 1
    try:
        next(g2)
    except StopIteration as e:
        print("return 值:", e.value)  # done

    # --- yield from：委托给子生成器 ---
    def inner():
        yield from range(3)

    print("yield from 结果:", list(inner()))  # [0, 1, 2]


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("1. 类实现迭代器 Countdown(3):", list(Countdown(3)))
    print("2. 生成器函数 countdown_gen(3):", list(countdown_gen(3)))
    print("=" * 50)
    gen_expr_demo()
    print("=" * 50)
    pipeline = only_even(read_lines(range(10)))
    print("4. 数据管道(偶数):", list(pipeline))
    print("=" * 50)
    one_shot_demo()
    print("=" * 50)
    tee_demo()
    print("=" * 50)
    infinite_demo()
    print("=" * 50)
    advanced_demo()
    print("=" * 50)
    print("全部示例运行完毕 ✔")


# 示例编号	演示内容	对应知识点
# 1	类实现迭代器	迭代协议 __iter__/__next__/StopIteration
# 2	生成器函数	yield 暂停—恢复机制
# 3	生成器表达式	惰性求值、圆括号语法
# 4	生成器管道	数据流式处理、低内存
# 5	一次性陷阱	耗尽后静默返回空（常见坑 1）
# 6	itertools.tee	多次遍历的正确解法
# 7	无限序列	惰性求值的典型应用
# 8	send/return/yield from	生成器高级特性（常见坑 2：return 值进入 StopIteration.value）
