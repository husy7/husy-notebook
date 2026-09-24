
## 1. os —— 操作系统与文件路径

### 1.1 工作目录与目录操作

```python
import os

os.getcwd()                 # 获取当前工作目录
os.chdir("/tmp")            # 切换工作目录
os.listdir(".")             # 列出目录内容（返回文件名列表）
os.mkdir("test")            # 创建单级目录（已存在会报错）
os.makedirs("a/b/c", exist_ok=True)   # 递归创建多级目录
os.rmdir("test")            # 删除空目录
os.removedirs("a/b/c")      # 递归删除空目录
```

### 1.2 文件操作

```python
os.remove("a.txt")          # 删除文件
os.rename("old.txt", "new.txt")        # 重命名/移动
os.replace("a.txt", "b.txt")           # 替换（目标存在也会覆盖，原子操作）
os.stat("a.txt")            # 文件状态：大小、修改时间等
os.stat("a.txt").st_size    # 文件字节数
os.stat("a.txt").st_mtime   # 最后修改时间戳
```

### 1.3 os.path —— 路径处理（高频）

```python
import os.path as osp

osp.join("data", "train", "img.jpg")   # 'data/train/img.jpg'（自动适配分隔符）
osp.exists("a.txt")         # 路径是否存在
osp.isfile("a.txt")         # 是否是文件
osp.isdir("data")           # 是否是目录
osp.isabs("/data/a.txt")    # 是否是绝对路径
osp.abspath("a.txt")        # 转绝对路径
osp.dirname("/a/b/c.txt")   # '/a/b'
osp.basename("/a/b/c.txt")  # 'c.txt'
osp.split("/a/b/c.txt")     # ('/a/b', 'c.txt')
osp.splitext("c.txt")       # ('c', '.txt')
osp.getsize("a.txt")        # 文件大小
osp.getmtime("a.txt")       # 修改时间戳（配合 datetime 转可读时间）
```

### 1.4 遍历目录树

```python
for dirpath, dirnames, filenames in os.walk("data"):
    for fn in filenames:
        full = os.path.join(dirpath, fn)
        print(full)
```

### 1.5 环境变量与系统信息

```python
os.environ.get("PATH")          # 读环境变量
os.environ["API_KEY"] = "xxx"   # 设环境变量（仅当前进程）
os.name          # 'posix' 或 'nt'
os.cpu_count()   # CPU 核数
os.getpid()      # 当前进程号
```

---

## 2. sys —— 与解释器交互

```python
import sys

# 命令行参数：python script.py a b  →  sys.argv == ['script.py', 'a', 'b']
first = sys.argv[1] if len(sys.argv) > 1 else None

sys.exit(0)          # 正常退出（0 表示成功，非 0 表示异常）
sys.exit("错误信息")  # 退出并打印信息到 stderr

sys.path             # 模块搜索路径列表
sys.path.append("/my/modules")   # 临时添加自定义模块路径

sys.version          # Python 版本信息字符串
sys.version_info     # 版本元组，可比较：sys.version_info >= (3, 9)
sys.platform         # 'win32' / 'linux' / 'darwin'
sys.maxsize          # 最大整数值

sys.getsizeof([1, 2, 3])         # 对象占用的字节数
sys.getrecursionlimit()          # 递归深度上限（默认约 1000）

# 输出重定向
sys.stdout.write("hello\n")      # 等价 print
print("error", file=sys.stderr)  # 错误输出

sys.modules          # 已导入模块的字典 {模块名: 模块对象}
sys.modules.keys()
```

---

## 3. math —— 数学函数

```python
import math

# 常量
math.pi          # 3.14159...
math.e           # 2.71828...
math.inf         # 无穷大（float('inf') 等价）
math.nan         # 非数

# 取整与绝对值（注意负数行为）
math.ceil(3.2)    # 4   向 +∞ 取整
math.ceil(-3.2)   # -3
math.floor(3.8)   # 3   向 -∞ 取整
math.floor(-3.8)  # -4
math.trunc(3.9)   # 3   向 0 截断
math.trunc(-3.9)  # -3
math.fabs(-3)     # 3.0（返回 float）

# 幂与对数
math.sqrt(16)     # 4.0
math.pow(2, 10)   # 1024.0（返回 float）
math.exp(1)       # e 的 1 次方
math.log(math.e)  # 1.0（自然对数）
math.log(100, 10) # 2.0（指定底数）
math.log2(8)      # 3.0
math.log10(1000)  # 3.0

# 数论
math.gcd(12, 18)      # 6  最大公约数
math.lcm(4, 6)        # 12 最小公倍数（3.9+）
math.factorial(5)     # 120

# 三角函数（弧度制！）
math.radians(180)     # π  角度转弧度
math.degrees(math.pi) # 180.0 弧度转角度
math.sin(math.pi / 2) # 1.0
math.cos(0)           # 1.0
math.atan2(1, 1)      # π/4  双参反正切，处理象限

# 其他常用
math.fmod(7, 3)       # 1.0（float 取模，与 % 对负数处理不同）
math.isnan(math.nan)  # True
math.isinf(math.inf)  # True
math.hypot(3, 4)      # 5.0（勾股定理，3.8+ 支持多维）
math.comb(10, 3)      # 120 组合数 C(10,3)
math.perm(10, 3)      # 720 排列数 A(10,3)
```

---

## 4. random —— 伪随机数

```python
import random

random.seed(42)   # 设置种子 → 结果可复现（调试必用）

# 随机实数
random.random()       # [0.0, 1.0) 均匀分布
random.uniform(1, 10) # [1, 10] 均匀分布浮点
random.gauss(0, 1)    # 正态分布（均值 0，标准差 1）
random.expovariate(1) # 指数分布

# 随机整数
random.randint(1, 10)   # [1, 10] 闭区间，含两端
random.randrange(10)    # [0, 10) 即 0~9
random.randrange(1, 10, 2)  # [1,10) 步长2：1,3,5,7,9

# 序列操作（重点）
lst = ['a', 'b', 'c', 'd', 'e']
random.choice(lst)          # 随机选一个
random.choices(lst, k=3)    # 有放回抽 3 个（可重复）
random.choices(lst, weights=[5,3,2,1,1], k=3)  # 按权重抽取
random.sample(lst, 3)       # 无放回抽 3 个（不重复，k≤len）
random.sample(range(100), 5)  # 配合 range 很好用
random.shuffle(lst)         # 原地打乱列表，返回 None！
```

**易错点总结：**
- `shuffle` 返回 `None`，不能写 `lst = random.shuffle(lst)`
- `sample` 的 k 不能大于序列长度，会抛 `ValueError`
- `choices` 可以指定 `weights` 做加权随机，常用于抽奖/加权采样
- 生产密码等安全场景用 `secrets` 模块，不用 `random`

---

## 5. datetime —— 日期与时间

```python
from datetime import datetime, date, time, timedelta

# 获取当前时间
now = datetime.now()        # 2026-09-24 14:30:00.123456
today = date.today()        # 2026-09-24
now.date()                  # 提取日期部分
now.time()                  # 提取时间部分

# 构造时间
dt = datetime(2026, 9, 24, 14, 30, 0)   # 注意月从 1 开始
d = date(2026, 9, 24)
t = time(14, 30, 0)

# 时间戳互转
now.timestamp()                    # → 1758695400.0（float 秒）
datetime.fromtimestamp(1758695400)   # 时间戳 → 本地 datetime
datetime.utcfromtimestamp(1758695400) # → UTC 时间（已弃用，用 fromtimestamp(ts, tz=timezone.utc)）

# 时间加减（timedelta）
tomorrow = now + timedelta(days=1)
last_week = now - timedelta(weeks=1)
future = now + timedelta(hours=3, minutes=30, seconds=15)
delta = tomorrow - now          # 结果是 timedelta
delta.days          # 天数（整数部分）
delta.seconds       # 不足一天的秒数
delta.total_seconds()  # 总秒数（float，最常用）

# 格式化：datetime → 字符串
now.strftime("%Y-%m-%d %H:%M:%S")   # '2026-09-24 14:30:00'
now.strftime("%Y年%m月%d日")         # '2026年09月24日'
# 常用格式码：%Y 四位年 %y 两位年 %m 月 %d 日
#            %H 时(24) %I 时(12) %M 分 %S 秒 %A 星期全称 %p AM/PM

# 解析：字符串 → datetime
datetime.strptime("2026-09-24", "%Y-%m-%d")          # 返回 datetime
datetime.strptime("24/09/2026 14:30", "%d/%m/%Y %H:%M")

# 比较与替换
d1 = date(2026, 1, 1); d2 = date(2026, 12, 31)
d1 < d2                  # True，可直接比较
dt.replace(year=2027, hour=0)  # 替换部分字段，返回新对象

# 常用属性
now.year, now.month, now.day, now.hour, now.minute, now.second
now.weekday()   # 周一=0 ... 周日=6
now.isoweekday() # 周一=1 ... 周日=7
```

**易错点：**
- `strptime` 格式串必须和字符串完全匹配，不匹配抛 `ValueError`
- `datetime` 既是模块名又是类名，`from datetime import datetime` 可避免歧义
- 格式化用的是 `strftime`（f = format），解析用的是 `strptime`（p = parse）

---

## 6. json —— 数据序列化

```python
import json

data = {
    "name": "小明",
    "age": 25,
    "scores": [90, 85, 77],
    "active": True,
    "note": None
}

# 内存中：对象 ↔ 字符串
s = json.dumps(data)                  # dict → JSON 字符串
obj = json.loads(s)                   # JSON 字符串 → dict

# 文件中：对象 ↔ 文件
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f)
with open("data.json", "r", encoding="utf-8") as f:
    obj = json.load(f)

# 常用参数
json.dumps(data, indent=2)            # 缩进美化输出
json.dumps(data, ensure_ascii=False)  # 中文不转义（输出"小明"而非"\u5c0f\u660e"）
json.dumps(data, sort_keys=True)      # key 按字母排序
json.dumps(data, separators=(",", ":"))  # 紧凑格式（省流量）
json.loads(s)                         # 非法 JSON 抛 json.JSONDecodeError

# 自定义对象序列化（datetime / set 不能直接转）
def default_handler(o):
    if isinstance(o, datetime):
        return o.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(o, set):
        return list(o)
    raise TypeError(f"不可序列化: {type(o)}")

json.dumps({"time": datetime.now()}, default=default_handler)
```

**易错点：**
- Python 与 JSON 类型映射：`True→true`、`False→false`、`None→null`、tuple→list
- 反序列化后所有数字都变成 `int` 或 `float`，元组变成列表
- `load/dump` 操作文件对象，`loads/dumps` 操作字符串（记忆：**s = string**）
- 读取的 JSON 必须符合规范：key 必须双引号、不允许尾逗号

---

## 综合小例子：把 5 个库串起来

统计当前目录所有 `.py` 文件的修改时间，结果写入 JSON：

```python
import os, sys, json
from datetime import datetime

result = []
for fn in os.listdir("."):
    if fn.endswith(".py"):
        mtime = os.path.getmtime(fn)
        result.append({
            "file": fn,
            "size": os.path.getsize(fn),
            "modified": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        })

result.sort(key=lambda x: x["modified"], reverse=True)

with open("report.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"共统计 {len(result)} 个文件", file=sys.stderr)
```

