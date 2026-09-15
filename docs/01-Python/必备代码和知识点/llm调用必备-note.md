
## 一、最小可用骨架

```python
import os, httpx

def call_llm(prompt, model, base_url, api_key=None, timeout=60):
    api_key = api_key or os.getenv("LLM_API_KEY")
    resp = httpx.post(
        base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
```

**四要素**：`URL` + `Header(Key)` + `Payload(model/messages)` + `解析路径`。

---

## 二、必记的 6 个知识点

### 1. 请求结构（OpenAI 兼容格式）
```python
{
    "model": "xxx",
    "messages": [
        {"role": "system",    "content": "你是助手"},
        {"role": "user",      "content": "问题"},
        {"role": "assistant", "content": "历史回答"},  # 多轮对话
    ],
    "temperature": 0.0,     # 0=确定，1=发散
    "max_tokens": 1024,     # 可选
    "reasoning_effort": "none", #是否思考，可选
}
```

### 2. 返回解析路径（记死这个）
```
resp.json()["choices"][0]["message"]["content"]
```
其他字段：`["usage"]["total_tokens"]`、`["finish_reason"]`

### 3. 认证头两种写法
```python
# OpenAI / DeepSeek / 通义 / Kimi / GLM
{"Authorization": f"Bearer {api_key}"} # Bearer后面有一个空格

# Anthropic Claude（特殊）
{"x-api-key": api_key, "anthropic-version": "2023-06-01"}
```

### 4. URL 规律
```
厂商域名 + /v1/chat/completions
```
> Claude 例外：`/v1/messages`

### 5. 异常必抓这几类
```python
except httpx.HTTPStatusError  # 4xx/5xx，看 exc.response.text
except httpx.TimeoutException # 超时
except httpx.HTTPError        # 连接失败（父类）
except (KeyError, IndexError, ValueError)  # 响应格式不对
```

### 6. Key 来源
```python
os.getenv("LLM_API_KEY")   # ✅ 永远不要硬编码
```

---

## 三、必记的 3 个"防坑"模式

### ① 重试 + 退避
```python
for attempt in range(retries + 1):
    try:
        ...
        return result
    except Exception as exc:
        if attempt == retries:
            raise
        time.sleep(0.5 * 2 ** attempt)  # 0.5s, 1s, 2s
```

### ② 打印错误体（调试 404/422 必备）
```python
resp = httpx.post(...)
if resp.status_code >= 400:
    print(resp.status_code, resp.text)  # ← 真相都在这
resp.raise_for_status()
```

### ③ 环境变量读取
```python
from dotenv import load_dotenv; load_dotenv()
# .env 里写 LLM_API_KEY=sk-xxx，代码里 os.getenv 读
```

---

## 四、常见状态码速查（必背）

| 码 | 含义 | 90% 的原因 |
|---|---|---|
| 401 | 未认证 | Key 没传 / 传错位置 |
| 403 | 拒绝 | Key 无效 / 欠费 / 地区限制 |
| **404** | 找不到 | **URL 少了 `/v1` 或模型名不对** |
| 422 | 参数错 | payload 多了不支持的字段 |
| 429 | 限流 | 加退避 |
| 500/502 | 服务端 | 重试 |

---

## 五、多轮对话（必记）

```python
messages = [{"role": "system", "content": "你是助手"}]

def chat(user_input):
    messages.append({"role": "user", "content": user_input})
    resp = httpx.post(base_url, headers=headers,
                      json={"model": model, "messages": messages})
    reply = resp.json()["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})  # ← 关键：存回历史
    return reply
```

---

## 六、流式输出（SSE，选记）

```python
with httpx.stream("POST", base_url, headers=headers, json={..., "stream": True}) as r:
    for line in r.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"].get("content", "")
            print(delta, end="", flush=True)
```

---

## 🎯 一句话总结必记核心

> **`headers` 带 Key → `json` 带 `model+messages` → `raise_for_status` → 取 `choices[0].message.content` → 用 `os.getenv` 存 Key → 失败看 `resp.text`。**
