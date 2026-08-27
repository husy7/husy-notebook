"""
综合案例：数据库会话与连接池管理器
涵盖知识点：
- 类实现 __enter__ / __exit__（会话管理器）
- contextlib.contextmanager 装饰器（连接工厂）
- contextlib.ExitStack 动态管理多个资源
- __exit__ 异常吞没与传播
- __enter__ 抛异常时 __exit__ 不调用
- 生成器上下文只能 yield 一次
- 异步上下文管理器 __aenter__ / __aexit__
- 与 try-finally 的对比（隐式 vs 显式）
- 自我检验问题验证
"""

import time
import asyncio
from contextlib import contextmanager, ExitStack, asynccontextmanager
from typing import Any, Dict, Optional


# ---------- 1. 类式上下文管理器：数据库会话 ----------
class DBSession:
    """模拟数据库会话，管理连接和事务"""

    def __init__(self, conn_id: str, auto_commit: bool = True):
        self.conn_id = conn_id
        self.auto_commit = auto_commit
        self._connected = False
        self._transaction_active = False

    def __enter__(self):
        """进入时建立连接并开启事务"""
        print(f"[{self.conn_id}] 正在建立连接...")
        # 模拟连接可能失败（演示 __enter__ 抛异常）
        if self.conn_id == "bad":
            raise ConnectionError(f"无法连接到 {self.conn_id}")
        self._connected = True
        self._transaction_active = True
        print(f"[{self.conn_id}] 连接成功，事务已开启")
        return self  # as 绑定的就是此对象

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时根据异常状态提交或回滚，并关闭连接"""
        if exc_type is not None:
            # 发生异常：回滚事务
            print(f"[{self.conn_id}] 检测到异常 ({exc_type.__name__})，执行回滚")
            self._transaction_active = False
        else:
            # 正常退出：提交事务（如果 auto_commit）
            if self.auto_commit and self._transaction_active:
                print(f"[{self.conn_id}] 正常结束，提交事务")
                self._transaction_active = False
            else:
                print(f"[{self.conn_id}] 正常结束，未提交（auto_commit=False）")

        # 关闭连接（模拟）
        self._connected = False
        print(f"[{self.conn_id}] 连接已关闭")

        # 返回 False（默认）表示不吞异常，异常继续向外抛
        # 若改为 True 则会吞掉异常，此处故意演示边界
        return False

    def execute(self, sql: str) -> str:
        """模拟执行SQL"""
        if not self._connected:
            raise RuntimeError("会话已关闭，无法执行")
        if not self._transaction_active:
            raise RuntimeError("事务已结束，无法执行")
        print(f"[{self.conn_id}] 执行: {sql}")
        return f"结果 of {sql}"


# ---------- 2. 装饰器式上下文管理器：连接工厂 ----------
@contextmanager
def create_connection(conn_id: str, auto_commit: bool = True):
    """
    使用 @contextmanager 创建连接，等同于 DBSession 的简化版
    演示 yield 必须包在 try/finally 中
    """
    print(f"[{conn_id}] (工厂) 正在连接...")
    # 模拟异常情况（连接失败）
    if conn_id == "broken":
        raise ConnectionError(f"连接 {conn_id} 拒绝访问")

    # 模拟连接对象（简单字典）
    conn = {"id": conn_id, "active": True}
    try:
        yield conn  # as 绑定的值
    except Exception as e:
        print(f"[{conn_id}] (工厂) 发生异常: {e}，执行回滚")
        # 这里可以回滚事务
        raise  # 重新抛出，不吞异常
    finally:
        # 清理代码必须放在 finally 中
        conn["active"] = False
        print(f"[{conn_id}] (工厂) 连接已关闭")


# ---------- 3. ExitStack 动态管理多个资源 ----------
def use_multiple_connections(conn_ids: list):
    """使用 ExitStack 动态管理不定数量的上下文"""
    with ExitStack() as stack:
        sessions = []
        for cid in conn_ids:
            # 动态进入上下文，并将清理函数注册到栈中
            session = stack.enter_context(DBSession(cid, auto_commit=True))
            sessions.append(session)
        # 现在可以同时使用多个会话
        for idx, session in enumerate(sessions):
            session.execute(f"SELECT * FROM users WHERE id={idx}")
        # 退出 with 时，栈会逆序（从右到左）调用每个会话的 __exit__
        # 注意：释放顺序与进入顺序相反


# ---------- 4. 异步上下文管理器（演示） ----------
class AsyncDBConnection:
    """异步上下文管理器，模拟 async with"""

    async def __aenter__(self):
        print("[Async] 异步连接建立")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("[Async] 异步连接释放")
        await asyncio.sleep(0.1)
        return False  # 不吞异常

    async def query(self, sql):
        await asyncio.sleep(0.05)
        return f"异步查询结果: {sql}"


# 也可以用 asynccontextmanager 装饰器
@asynccontextmanager
async def async_conn_factory():
    print("[Async工厂] 连接中...")
    try:
        yield {"status": "ready"}
    finally:
        print("[Async工厂] 释放中...")


# ---------- 5. 演示各种场景 ----------
def demo_basic():
    """基本用法：正常流程"""
    print("\n===== 1. 基本用法（正常流程） =====")
    with DBSession("sess1") as sess:
        sess.execute("INSERT INTO logs VALUES ('hello')")
    # 退出后自动提交并关闭


def demo_exception():
    """异常流程：__exit__ 中回滚，且异常继续传播"""
    print("\n===== 2. 异常流程（回滚 + 传播） =====")
    try:
        with DBSession("sess2") as sess:
            sess.execute("UPDATE accounts SET balance=balance-100")
            raise ValueError("余额不足！")  # 触发异常
    except ValueError as e:
        print(f"捕获到外部异常: {e}")


def demo_enter_fails():
    """__enter__ 抛异常时 __exit__ 不会被调用"""
    print("\n===== 3. __enter__ 失败（__exit__ 不调用） =====")
    try:
        with DBSession("bad") as sess:  # "bad" 触发 __enter__ 抛 ConnectionError
            sess.execute("永远不会执行")
    except ConnectionError as e:
        print(f"捕获连接异常: {e}")


def demo_contextmanager_deco():
    """使用 @contextmanager 装饰器"""
    print("\n===== 4. @contextmanager 方式 =====")
    with create_connection("deco1") as conn:
        print(f"使用连接: {conn}")
        # 模拟异常
        # raise RuntimeError("模拟错误")
    # 即使异常，finally 也会执行


def demo_exitstack():
    """动态管理多个资源（ExitStack）"""
    print("\n===== 5. ExitStack 动态管理多个会话 =====")
    use_multiple_connections(["a", "b", "c"])


def demo_async():
    """异步上下文管理器（需要 asyncio 运行）"""
    print("\n===== 6. 异步上下文管理器 =====")

    async def async_main():
        async with AsyncDBConnection() as conn:
            result = await conn.query("SELECT 1")
            print(result)
        # 使用 asynccontextmanager
        async with async_conn_factory() as factory:
            print(f"工厂连接: {factory}")

    asyncio.run(async_main())


def demo_common_pitfalls():
    """演示常见坑"""
    print("\n===== 7. 常见坑演示 =====")

    # 坑1：在 with 块外使用资源（会报错）
    sess = DBSession("pitfall")
    with sess:
        sess.execute("合法操作")
    # sess.execute("非法操作")  # 取消注释会报 RuntimeError: 会话已关闭

    # 坑2：__exit__ 返回 True 吞异常（默认 False，此处不演示，但强调）
    class Suppressor:
        def __enter__(self): return self
        def __exit__(self, *args): return True  # 吞掉所有异常

    with Suppressor():
        raise ValueError("这个异常会被吞掉，不会传播")
    print("程序继续运行，异常被静默吞没！—— 这是危险行为")

    # 坑3：生成器上下文 yield 放在 try 外面（错误示例，但这里不实际执行）
    # @contextmanager
    # def bad_generator():
    #     print("enter")
    #     yield
    #     print("exit")   # 若 yield 抛异常，这行不会执行，释放失败


def demo_comparison_with_try_finally():
    """对比 try-finally 手动清理（显式 vs 隐式）"""
    print("\n===== 8. 对比 try-finally（隐式清理 vs 显式清理） =====")
    # 传统方式
    conn = None
    try:
        conn = DBSession("manual")
        conn.__enter__()
        conn.execute("手工操作")
    finally:
        if conn is not None:
            conn.__exit__(None, None, None)  # 需要手动调用，容易遗漏
    # 上下文管理器方式更简洁、不易遗漏


# ---------- 主程序 ----------
if __name__ == "__main__":
    demo_basic()
    demo_exception()
    demo_enter_fails()
    demo_contextmanager_deco()
    demo_exitstack()
    demo_async()
    demo_common_pitfalls()
    demo_comparison_with_try_finally()

    # 自我检验问题的答案体现在代码中：
    # 1. 一句话说清：保证资源在任何退出路径（包括异常）下正确释放。
    # 2. 最小示例：类实现 __enter__/__exit__ 或 @contextmanager+try/finally+yield。
    # 3. 常见错误：__exit__ 返回 True 吞异常；__enter__ 抛异常时 __exit__ 不调用；生成器 yield 后不能继续。
    # 4. 与 try-finally 区别：清理逻辑封装进对象，由 with 自动触发，可复用且不可能被遗忘。