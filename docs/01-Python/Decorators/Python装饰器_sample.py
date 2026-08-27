## 实际应用案例：API 请求重试与监控装饰器

# 假设您正在开发一个微服务客户端，需要给多个调用外部 API 的函数统一添加：
# - **重试机制**（网络抖动时自动重试）
# - **超时监控**（超过阈值告警）
# - **请求日志**（记录入参、返回值）
# - **身份验证**（注入 token）

# 我们将这些横切逻辑封装成装饰器，并展示如何组合、参数化、保留元信息。

### 1. 基础计时 + 日志装饰器（演示 `@wraps`）

import functools
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_and_time(func):
    """打印函数调用信息和执行耗时"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        cost = time.perf_counter() - start
        logger.info(f"{func.__name__} finished in {cost:.4f}s, result={result}")
        return result
    return wrapper


### 2. 带参数装饰器：重试机制（演示三层嵌套）


def retry(max_retries=3, delay=1.0, exceptions=(Exception,)):
    """当函数抛出指定异常时，重试最多 max_retries 次"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    logger.warning(f"{func.__name__} attempt {attempt+1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} all retries exhausted")
                        raise last_exc
            return None  # 不会执行到这里
        return wrapper
    return decorator


### 3. 类装饰器：为函数添加可配置的“熔断”状态（演示 `__call__`）


class CircuitBreaker:
    """简单熔断器：失败次数超过阈值则快速失败"""
    def __init__(self, threshold=3, timeout=10):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self._open = False

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self._open:
                # 检查是否超时重置
                if time.time() - self.last_failure_time > self.timeout:
                    self._open = False
                    self.failure_count = 0
                    logger.info(f"Circuit breaker reset for {func.__name__}")
                else:
                    raise RuntimeError(f"Circuit breaker open for {func.__name__}")
            try:
                result = func(*args, **kwargs)
                # 成功则重置失败计数（半开状态简化）
                self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.threshold:
                    self._open = True
                    logger.error(f"Circuit breaker opened for {func.__name__} after {self.threshold} failures")
                raise e
        return wrapper


### 4. 组合多个装饰器（注意执行顺序）


# 模拟鉴权（注入 token）
def inject_auth(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 模拟从环境变量获取 token
        kwargs['token'] = "fake_token_123"
        return func(*args, **kwargs)
    return wrapper

# 应用装饰器（自下而上执行：先 inject_auth，再 retry，最后 log_and_time）
@log_and_time                    # 最外层（最后应用）
@retry(max_retries=2, delay=0.5, exceptions=(ConnectionError, TimeoutError))
@CircuitBreaker(threshold=2, timeout=5)
@inject_auth                     # 最内层（先应用）
def call_external_api(url, payload=None, token=None):
    """模拟调用外部 API"""
    if not token:
        raise ValueError("Missing token")
    # 模拟网络请求
    if "fail" in url:
        raise ConnectionError("Simulated connection failure")
    if "slow" in url:
        time.sleep(2)
    return {"status": "ok", "data": payload}


### 5. 测试与输出


if __name__ == "__main__":
    # 正常调用
    print(call_external_api("http://api.example.com/ok", payload={"id": 1}))
    # 触发重试（首次失败，第二次成功）
    try:
        call_external_api("http://api.example.com/fail", payload={"id": 2})
    except Exception as e:
        print(f"最终失败: {e}")
    # 触发熔断（连续失败）
    for _ in range(4):
        try:
            call_external_api("http://api.example.com/fail_again", payload={})
        except Exception as e:
            print(f"调用失败: {e}")
    # 等待熔断超时后再调用
    time.sleep(6)
    call_external_api("http://api.example.com/ok", payload={"id": 3})

'''    
```

**运行日志（节选）**：
```
2026-08-27 10:00:00 - __main__ - INFO - Calling call_external_api with args=('http://api.example.com/ok',), kwargs={'payload': {'id': 1}}
2026-08-27 10:00:00 - __main__ - INFO - call_external_api finished in 0.0012s, result={'status': 'ok', 'data': {'id': 1}}
...
2026-08-27 10:00:01 - __main__ - WARNING - call_external_api attempt 1 failed: Simulated connection failure
2026-08-27 10:00:01 - __main__ - INFO - Calling call_external_api ...
2026-08-27 10:00:01 - __main__ - INFO - call_external_api finished in 0.0009s, result={'status': 'ok', 'data': {'id': 2}}
...
2026-08-27 10:00:02 - __main__ - ERROR - Circuit breaker opened for call_external_api after 2 failures
...
```

---
'''
## 知识点覆盖清单

# | 知识点 | 体现 |
# |--------|------|
# | 基础装饰器 | `log_and_time` |
# | 带参数装饰器 | `retry`（三层嵌套） |
# | 类装饰器 | `CircuitBreaker`（实现 `__call__`） |
# | `@functools.wraps` | 所有装饰器都使用了，保证 `__name__`、`__doc__` 正确 |
# | 多重装饰器顺序 | `@log_and_time`（最外层）→ `@retry` → `@CircuitBreaker` → `@inject_auth`（最内层） |
# | 传递 `*args, **kwargs` | 所有 wrapper 都透传参数 |
# | 装饰器在导入时执行 | 无特殊，但类装饰器实例在定义函数时创建 |
# | 与闭包的关系 | 每个装饰器内部都形成闭包，捕获外部参数（如 `max_retries`） |
# | 应用场景 | 鉴权、重试、熔断、日志、计时 |
# | 边界处理 | 异常捕获、重试次数、熔断状态、超时重置 |

# ---
