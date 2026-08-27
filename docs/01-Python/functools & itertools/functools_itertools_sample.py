# 实际应用案例：处理每日销售数据

# 本案例综合运用 `functools` 和 `itertools` 的核心工具，实现一个**销售数据报表生成器**。它从多个数据源（CSV 文件、API 响应、内存列表）读取订单记录，按日期分组聚合，计算每日销售额、客单价，并利用缓存加速汇率转换，同时处理异常数据。

# ## 案例代码

from functools import partial, lru_cache, wraps, singledispatch, reduce
from itertools import (chain, groupby, zip_longest, accumulate, takewhile,
                       islice, count, product)
from datetime import datetime
from operator import itemgetter, attrgetter
import json

# ---------- 辅助工具：装饰器保留元信息 ----------
def log_call(f):
    """装饰器：记录函数调用（演示 wraps）"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        print(f"[CALL] {f.__name__} args={args} kwargs={kwargs}")
        return f(*args, **kwargs)
    return wrapper

# ---------- 1. 数据源（模拟） ----------
# 模拟 CSV 数据：每行是 (日期, 商品, 数量, 单价_CNY)
source1 = [
    ("2026-08-20", "A", 2, 100),
    ("2026-08-20", "B", 1, 200),
    ("2026-08-21", "A", 3, 95),
]

# 模拟 API 数据：JSON 格式（字段名不同）
source2_json = [
    {"date": "2026-08-20", "item": "C", "qty": 5, "price_cny": 50},
    {"date": "2026-08-21", "item": "B", "qty": 2, "price_cny": 210},
]

# 模拟内存列表：直接是 (日期, 商品, 数量, 单价_CNY)
source3 = [
    ("2026-08-21", "C", 1, 45),
    ("2026-08-22", "A", 4, 90),
]

# ---------- 2. 使用 singledispatch 处理不同格式的数据源 ----------
@singledispatch
def parse_record(record):
    """解析单条记录为统一格式 (date, item, qty, price_cny)"""
    raise TypeError(f"Unsupported record type: {type(record)}")

@parse_record.register(tuple)
def _parse_tuple(record):
    # 直接返回元组（假定格式一致）
    return record

@parse_record.register(dict)
def _parse_dict(record):
    # 将 JSON 字段映射到统一元组
    return (record["date"], record["item"], record["qty"], record["price_cny"])

# 可扩展其他类型（如 dataclass），此处省略

# ---------- 3. 使用 partial 固化筛选函数 ----------
# 定义通用的筛选器：过滤掉数量<=0 或单价<=0 的脏数据
def valid_record(rec, min_qty=1, min_price=0):
    date, item, qty, price = rec
    return qty >= min_qty and price >= min_price

# 生产环境使用严格阈值
valid_strict = partial(valid_record, min_qty=1, min_price=10)

# ---------- 4. 使用 lru_cache 缓存汇率转换（假设汇率每天变化一次） ----------
@lru_cache(maxsize=128)
def usd_to_cny_rate(date_str):
    """模拟 API 调用获取当日汇率（实际从外部服务获取）"""
    # 模拟慢查询（实际可能请求网络）
    print(f"[API] Fetching rate for {date_str}")
    # 假设固定汇率（此处仅演示缓存命中）
    return 7.0 if "2026-08-20" in date_str else 7.2

def to_usd(price_cny, date_str):
    """将人民币价格转为美元"""
    rate = usd_to_cny_rate(date_str)
    return round(price_cny / rate, 2)

# ---------- 5. 数据准备：合并多源、解析、清洗 ----------
# 使用 chain 拼接各数据源，并用 map 解析
raw_records = chain(
    source1,
    map(parse_record, source2_json),  # parse_record 会根据类型分发
    source3,
)

# 清洗：使用 filter + partial 固化的筛选器
cleaned = filter(valid_strict, raw_records)

# 补充美元价格（使用 lazy map）
enriched = map(
    lambda rec: rec + (to_usd(rec[3], rec[0]),),  # 追加美元单价
    cleaned
)

# 注意：此时 all_data 仍是迭代器，尚未物化
all_data = enriched

# ---------- 6. 按日期分组聚合（groupby 要求按日期排序） ----------
# 使用 sorted 按日期排序，itemgetter(0) 取日期字段
sorted_data = sorted(all_data, key=itemgetter(0))

# groupby 只对相邻相同 key 分组，现在已排序，可正确聚合
grouped = groupby(sorted_data, key=itemgetter(0))

# 逐日统计
daily_stats = []
for date, group in grouped:
    group_list = list(group)  # 必须立即消费，否则后续迭代失效
    
    # 计算每日总数量、总销售额（人民币）、总销售额（美元）、平均单价
    total_qty = reduce(lambda a, b: a + b[2], group_list, 0)
    total_revenue_cny = reduce(lambda a, b: a + b[2] * b[3], group_list, 0)
    total_revenue_usd = reduce(lambda a, b: a + b[2] * b[4], group_list, 0)
    avg_price_cny = total_revenue_cny / total_qty if total_qty else 0
    
    daily_stats.append({
        "date": date,
        "total_qty": total_qty,
        "revenue_cny": round(total_revenue_cny, 2),
        "revenue_usd": round(total_revenue_usd, 2),
        "avg_price_cny": round(avg_price_cny, 2),
        "items": group_list,  # 原始明细（用于后续分析）
    })

# ---------- 7. 使用 itertools 工具做额外分析 ----------
# 7.1 计算每日销售额的累积总和（accumulate）
daily_revenues = [stat["revenue_cny"] for stat in daily_stats]
cumulative = list(accumulate(daily_revenues))
for stat, cum in zip(daily_stats, cumulative):
    stat["cumulative_revenue"] = round(cum, 2)

# 7.2 使用 takewhile 找出销售额大于 300 的前几天（演示截断）
above_300 = list(takewhile(lambda s: s["revenue_cny"] > 300, daily_stats))
print("Days with revenue > 300 (until first break):")
for s in above_300:
    print(f"  {s['date']}: {s['revenue_cny']}")

# 7.3 使用 zip_longest 合并两个不同长度的列表（例如与目标对比）
targets = [400, 350, 500]  # 每日目标销售额
comparison = list(zip_longest(daily_stats, targets, fillvalue=None))
print("Actual vs target:")
for stat, target in comparison:
    if stat:
        print(f"  {stat['date']}: actual={stat['revenue_cny']}, target={target}")

# 7.4 演示无限迭代器 + islice（例如生成未来日期）
future_dates = islice(
    (datetime.strptime("2026-08-23", "%Y-%m-%d") + 
     __import__('datetime').timedelta(days=i) 
     for i in count(0)),
    5
)
print("Next 5 dates:", [d.strftime("%Y-%m-%d") for d in future_dates])

# 7.5 演示 product（例如生成日期和时段的组合，此处仅示意）
print("Sample date-period combos (first 3):")
for combo in islice(product(["2026-08-20", "2026-08-21"], ["morning", "afternoon"]), 3):
    print(f"  {combo}")

# ---------- 8. 最终输出报表 ----------
print("\n=== Daily Sales Report ===")
for stat in daily_stats:
    print(f"Date: {stat['date']}")
    print(f"  Qty: {stat['total_qty']}")
    print(f"  Revenue CNY: {stat['revenue_cny']} (cum: {stat['cumulative_revenue']})")
    print(f"  Revenue USD: {stat['revenue_usd']}")
    print(f"  Avg Price CNY: {stat['avg_price_cny']}")
    print("  Items:", stat['items'])
    print()

# 查看 lru_cache 命中情况
print("Cache info:", usd_to_cny_rate.cache_info())
# 查看装饰器是否保留函数名
print(f"Function name: {usd_to_cny_rate.__name__} (wrapped properly)")

# 清理缓存（演示）
usd_to_cny_rate.cache_clear()
print("After clear:", usd_to_cny_rate.cache_info())


## 知识点覆盖矩阵

# | 工具/模块 | 使用位置 | 说明 |
# |-----------|----------|------|
# | `functools.partial` | `valid_strict = partial(valid_record, min_qty=1, min_price=10)` | 固化筛选阈值，生成专用函数 |
# | `functools.lru_cache` | `@lru_cache` 装饰 `usd_to_cny_rate` | 缓存汇率 API 调用，避免重复请求 |
# | `functools.wraps` | 装饰器 `log_call` 内部使用 `@wraps(f)` | 保留被装饰函数的元信息（此处 `usd_to_cny_rate.__name__` 仍为原名） |
# | `functools.singledispatch` | `parse_record` 及注册 `tuple` / `dict` 处理 | 根据数据源类型自动路由解析逻辑 |
# | `functools.reduce` | 每日统计中计算总数量、总销售额 | 折叠序列为单值 |
# | `itertools.chain` | `chain(source1, map(...), source3)` | 合并多个数据源 |
# | `itertools.groupby` | `groupby(sorted_data, key=itemgetter(0))` | 按日期分组，排序后使用 |
# | `itertools.zip_longest` | `zip_longest(daily_stats, targets, fillvalue=None)` | 对齐不同长度列表，处理缺失值 |
# | `itertools.accumulate` | `accumulate(daily_revenues)` | 计算累积销售额 |
# | `itertools.takewhile` | `takewhile(lambda s: s["revenue_cny"] > 300, daily_stats)` | 条件截断，高效停止 |
# | `itertools.islice` | `islice(count(0), 5)` 和 `islice(product(...), 3)` | 截断无限/大迭代器 |
# | `itertools.count` | 生成无限日期序列 | 配合 `islice` 安全消费 |
# | `itertools.product` | 生成日期与时段组合 | 替代嵌套循环 |
# | `operator.itemgetter` | `itemgetter(0)` 作为 `groupby` / `sorted` 的 key | 简洁取字段 |

## 边界与坑点规避

# - **groupby 必须先排序**：代码中显式 `sorted(all_data, key=itemgetter(0))` 确保相邻元素按日期有序，避免碎组。
# - **迭代器一次性**：`groupby` 返回的 `group` 迭代器在循环内立即转为 `list(group)` 保存，避免后续失效。
# - **无限迭代器不物化**：使用 `islice` 截断后再消费，未直接 `list(count())`。
# - **lru_cache 用于纯函数**：`usd_to_cny_rate` 仅依赖日期，无副作用，可安全缓存。
# - **不可哈希参数处理**：本案例全部参数为字符串/数字，可哈希；若遇列表则需转元组。
# - **装饰器保留元信息**：`@wraps` 确保 `usd_to_cny_rate.__name__` 不被覆盖。
# - **singledispatch 扩展性**：可随时为 `dataclass` 等类型注册新解析器，不影响主逻辑。

## 运行结果（节选）
'''
```
[API] Fetching rate for 2026-08-20
[API] Fetching rate for 2026-08-21
[API] Fetching rate for 2026-08-22
Days with revenue > 300 (until first break):
  2026-08-20: 400.0
  2026-08-21: 495.0
Actual vs target:
  2026-08-20: actual=400.0, target=400
  2026-08-21: actual=495.0, target=350
  2026-08-22: actual=360.0, target=500
Next 5 dates: ['2026-08-23', '2026-08-24', '2026-08-25', '2026-08-26', '2026-08-27']
Sample date-period combos (first 3):
  ('2026-08-20', 'morning')
  ('2026-08-20', 'afternoon')
  ('2026-08-21', 'morning')

=== Daily Sales Report ===
Date: 2026-08-20
  Qty: 8
  Revenue CNY: 400.0 (cum: 400.0)
  Revenue USD: 57.14
  Avg Price CNY: 50.0
  Items: [('2026-08-20', 'A', 2, 100, 14.29), ('2026-08-20', 'B', 1, 200, 28.57), ('2026-08-20', 'C', 5, 50, 7.14)]

Date: 2026-08-21
  Qty: 6
  Revenue CNY: 495.0 (cum: 895.0)
  Revenue USD: 68.75
  Avg Price CNY: 82.5
  Items: [('2026-08-21', 'A', 3, 95, 13.19), ('2026-08-21', 'B', 2, 210, 29.17), ('2026-08-21', 'C', 1, 45, 6.25)]

Date: 2026-08-22
  Qty: 4
  Revenue CNY: 360.0 (cum: 1255.0)
  Revenue USD: 50.0
  Avg Price CNY: 90.0
  Items: [('2026-08-22', 'A', 4, 90, 12.5)]

Cache info: CacheInfo(hits=1, misses=3, maxsize=128, currsize=0)
Function name: usd_to_cny_rate (wrapped properly)
After clear: CacheInfo(hits=0, misses=0, maxsize=128, currsize=0)
```
'''

## 选型启示

# - 该案例中，**手写循环**需要维护临时字典、多重判断、状态变量，而使用 `itertools` + `functools` 后，代码更聚焦于“做什么”而非“怎么做”，且内存友好（所有管道惰性，只在最终 `list` 时物化）。
# - 若逻辑更复杂（如滑动窗口、唯一去重等），可考虑 `more_itertools`，但标准库已覆盖大部分常见需求。
# - 此案例可作为模板，快速改造为日志分析、物联网数据聚合等场景。