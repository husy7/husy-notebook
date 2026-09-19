结论：FastAPI 基础不用“刷很多题”，核心是能**不看文档手写一个迷你 API**：路由 + Pydantic 校验 + Depends + 异常 + CRUD + 分页 + JWT + 测试。下面这些能默写，基础就够。

## P0：必须能手撕

### 1. Hello World + 路径/查询参数校验
```python
@app.get("/items/{item_id}")
def get_item(
    item_id: int,
    q: str | None = None,
    limit: int = Query(10, ge=1, le=100),
):
    return {"item_id": item_id, "q": q, "limit": limit}
```
考察：HTTP 方法、路径参数、查询参数、类型转换、`Query/Path` 校验。

### 2. Pydantic 请求体 + 响应模型
```python
class ItemCreate(BaseModel):
    name: str = Field(min_length=1)
    price: float = Field(gt=0)

class ItemOut(ItemCreate):
    id: int
    model_config = ConfigDict(from_attributes=True)  # v1 是 orm_mode=True

@app.post("/items", response_model=ItemOut, status_code=201)
def create_item(item: ItemCreate):
    ...
```
考察：请求体校验、响应过滤、敏感字段不返回、Pydantic v2 用法。

### 3. Depends 依赖注入
```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/me")
def read_me(user=Depends(get_current_user), db=Depends(get_db)):
    ...
```
考察：函数依赖、类依赖、子依赖、`yield` 清理、依赖缓存。

### 4. 异常处理与状态码
```python
raise HTTPException(status_code=404, detail="Item not found")

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
```
考察：`HTTPException`、自定义异常、全局异常处理器。

### 5. APIRouter 拆分路由
```python
router = APIRouter(prefix="/items", tags=["items"])

@router.get("/{item_id}")
def get_item(item_id: int):
    ...

app.include_router(router)
```
考察：项目结构、`prefix/tags/dependencies`。

### 6. SQLAlchemy CRUD + 分页
```python
@app.get("/items")
def list_items(
    skip: int = 0,
    limit: int = Query(10, le=100),
    db: Session = Depends(get_db),
):
    return db.query(Item).offset(skip).limit(limit).all()
```
考察：模型定义、Session 依赖、增删改查、分页、过滤。

### 7. JWT 登录 + 鉴权
```python
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate(form.username, form.password)
    if not user:
        raise HTTPException(401, "Incorrect username or password")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    ...
```
考察：密码哈希、OAuth2 表单、JWT 编解码、当前用户依赖。

### 8. TestClient 测试
```python
client = TestClient(app)

def test_create_item():
    r = client.post("/items", json={"name": "a", "price": 1})
    assert r.status_code == 201
```
考察：接口测试、依赖覆盖、状态码断言。

## P1：建议会写

9. CORS 与中间件  
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```
10. 文件上传 / 表单  
```python
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    return {"filename": file.filename, "size": len(content)}
```
11. BackgroundTasks  
12. lifespan 生命周期 + pydantic-settings 配置  
13. WebSocket echo  
14. 静态文件、模板、文件下载  
15. OpenAPI 文档定制：`tags/summary/response_description`

## P2：加分，不是基础必须

- 异步数据库：async SQLAlchemy + asyncpg  
- OAuth2 完整授权码、refresh token  
- 限流、缓存、Celery  
- 复杂权限：RBAC、多角色  
- 分页封装、统一响应格式  
- 依赖注入高级用法：`use_cache=False`、全局依赖

## 验收标准

如果你能在 40 分钟内，用 SQLite 手写一个迷你 API：

- 用户注册 / 登录
- JWT 鉴权
- 当前用户接口
- Todo 或 Item 的 CRUD
- 分页、404、400 异常
- CORS
- APIRouter 拆文件
- TestClient 测试

那 FastAPI 基础就够用了。否则先别刷偏门，按上面 P0 手撕三遍：  
**第一遍单文件无数据库，第二遍加 SQLite CRUD，第三遍加 JWT + 测试。**