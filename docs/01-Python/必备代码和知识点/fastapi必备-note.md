
## 一、最小可用骨架

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    tags: list[str] = []

@app.get("/")
def read_root():
    return {"hello": "world"}

@app.post("/items/")
def create_item(item: Item):
    return {"name": item.name, "price": item.price}

# 启动：uvicorn main:app --reload
```

**三要素**：`app = FastAPI()` + 装饰器 `@app.get/post` + 类型注解自动校验。

---

## 二、必记的 6 个知识点

### 1. 路径参数（Path）
```python
@app.get("/items/{item_id}")
def get_item(item_id: int):   # ← 类型注解自动转换 + 校验
    return {"item_id": item_id}
```
> `item_id: int` 会自动把 `/items/abc` 拦下来返回 422。

### 2. 查询参数（Query）
```python
@app.get("/items/")
def list_items(skip: int = 0, limit: int = 10, q: str | None = None):
    # 非路径中的参数 = 查询参数，带默认值 = 可选
    return {"skip": skip, "limit": limit, "q": q}
```

### 3. 请求体（Body）
```python
class Item(BaseModel):
    name: str
    price: float
    description: str | None = None   # 可选

@app.post("/items/")
def create(item: Item, important: bool = False):
    # item 是 body（Pydantic 模型），important 是 query
    return item
```

### 4. 三种参数区分（记死）
| 来源 | 写法 | 例子 |
|---|---|---|
| **路径** | 出现在 URL 模板里 `{x}` | `/items/{id}` |
| **查询** | 函数参数，非路径、非模型 | `?skip=0&limit=10` |
| **请求体** | 参数是 `BaseModel` 子类 | POST 的 JSON |

### 5. 状态码 & 错误
```python
from fastapi import HTTPException, status

@app.get("/items/{id}")
def get(id: int):
    if id not in db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found",
        )
    return db[id]

# 自定义成功状态码
@app.post("/items/", status_code=status.HTTP_201_CREATED)
def create(item: Item): ...
```

### 6. 自动文档（白送）
```
http://127.0.0.1:8000/docs     # Swagger UI
http://127.0.0.1:8000/redoc    # ReDoc
```

---

## 三、必记的 Pydantic 校验

```python
from pydantic import BaseModel, Field
from typing import Annotated

class Item(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)  # ... 表示必填
    price: float = Field(gt=0, description="必须大于 0")
    tags: list[str] = []

# 嵌套模型
class Order(BaseModel):
    item: Item                    # ← 嵌套
    quantity: int = 1

# 响应模型（过滤字段，比如隐藏密码）
@app.post("/user/", response_model=UserOut)  # 只返回 UserOut 里定义的字段
def create_user(user: UserIn): ...
```

**必记字段约束**：`min_length` / `max_length` / `gt` / `ge` / `lt` / `le` / `regex`（Pydantic v2 叫 `pattern`）

---

## 四、必记的 3 个「进阶但高频」

### ① 依赖注入（Depends）
```python
from fastapi import Depends

def get_db():
    db = SessionLocal()
    try:
        yield db       # ← yield 版依赖会自动清理
    finally:
        db.close()

@app.get("/users/")
def list_users(db = Depends(get_db)):
    return db.query(User).all()
```
> **用途**：数据库连接、登录校验、公共参数。**FastAPI 的灵魂功能。**

### ② 中间件（CORS 必备）
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### ③ 路由分组（APIRouter）
```python
from fastapi import APIRouter

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/")
def list_users(): ...

# main.py 里
app.include_router(router)
```

---

## 五、必记的启动 & 部署

```bash
# 开发（热重载）
uvicorn main:app --reload --port 8000

# 生产（多 worker）
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

```python
# main.py 里直接跑
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 六、常见坑速查

| 现象 | 原因 |
|---|---|
| 422 Unprocessable Entity | body/query 类型不符，看 `detail` 字段 |
| `Field required` | 必填字段没传 |
| 前端跨域报错 | 忘了加 `CORSMiddleware` |
| `Depends` 没生效 | 忘写 `Depends(...)`，直接写函数名了 |
| Pydantic v1/v2 报错 | `regex` → `pattern`，`Config` → `model_config` |
| 返回 ORM 对象报错 | 加 `from_attributes=True`（v2）或 `orm_mode=True`（v1） |
| async 里用同步阻塞 | `async def` 里跑了同步 IO，会堵事件循环 |

---

## 🎯 一句话总结必记核心

> **`app = FastAPI()` → 装饰器 `@app.get/post` → 类型注解 = 自动校验 → 路径/查询/请求体三来源 → `HTTPException` 报错 → `Depends` 复用逻辑 → `/docs` 自动文档。**

---

## 附：和 LLM 调用结合的最小示例

```python
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os, httpx

load_dotenv()
app = FastAPI()

class ChatIn(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(body: ChatIn):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            os.getenv("LLM_BASE_URL"),
            headers={"Authorization": f"Bearer {os.getenv('LLM_API_KEY')}"},
            json={"model": os.getenv("LLM_MODEL"),
                  "messages": [{"role": "user", "content": body.prompt}]},
            timeout=60,
        )
        resp.raise_for_status()
        return {"reply": resp.json()["choices"][0]["message"]["content"]}

# uvicorn main:app --reload
# 访问 /docs 直接测
```

**注意**：这里用了 `async def` + `httpx.AsyncClient`（异步版），别在 `async def` 里用同步 `httpx.post`，否则会堵事件循环。