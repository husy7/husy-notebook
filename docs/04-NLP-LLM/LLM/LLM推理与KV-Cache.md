---
title: "LLM 推理：KV-Cache 与采样策略"
tags: [LLM, 推理, KV-Cache, 采样, temperature, top-k, top-p]
date: 2026-08-29
---

# LLM 推理：KV-Cache 与采样策略

## 一、核心思想

LLM（Decoder-only Transformer）推理是**逐 token 自回归生成**：每次生成一个 token，再把新 token 拼进输入继续。两个关键工程/算法问题：

1. **怎么生成得又快又省？** → **KV-Cache**：缓存历史的 Key/Value，避免重复计算整段前缀。
2. **怎么从"下一个 token 的概率分布"里挑 token 显得自然、高质量？** → **采样策略**（greedy / temperature / top-k / top-p）。

## 二、KV-Cache：推理加速核心

### 2.1 朴素做法的问题

生成第 i 个 token 时，若每次都把**全部历史序列**重新过一遍模型，需要重算前面所有 token 的注意力 → 生成 n 个 token 复杂度 O(n³)，极慢。

### 2.2 KV-Cache 的做法

自注意力中，第 t 步只需要新的 Q，而 K/V 可用 **之前所有步骤缓存的 K/V** 拼接复用：

```
第一步: 算 K1,V1 → 存下来
第二步: 算 K2,V2 → 用 [K1,K2]/[V1,V2] 做注意力 → 存 [K1,K2,V1,V2]
...
```

- **效果**：每步只需算新 token 的 K/V → 复杂度降为 O(n²)，且省重复计算。
- **代价**：KV-Cache 随序列长度线性增长，**占显存**（长上下文 + 大 batch 时是显存瓶颈）。

```text
每生成一步：
   新 token → 计算新 K,V → 追加进 cache
   attention 用 cache 的全部 K,V 与当前 Q
```

### 2.3 显存优化方向

- KV 量化和缓存压缩（如 8-bit/4-bit cache）。
- 多查询/分组查询注意力（GQA/MQA）——共享 K/V，显著省显存（当代 LLM 标配）。
- MLA（DeepSeek）把 K/V 低秩压缩，进一步省缓存。
- 长上下文进一步用滑动窗口 + 全局 token。

## 三、采样策略：如何选下一个 token

模型输出一个词表大小的概率分布 $P(\text{token}_t | \text{context})$。

### 3.1 Greedy（贪心）

直接选概率最大的 token。`temperature=0` + 稳定，但易**重复、机械**，缺少创造性。

### 3.2 Temperature（温度）

把 logits 除以温度 T，再 softmax：

$$ P \propto \exp(\text{logit} / T) $$

| T | 效果 |
|----|------|
| T→0 | 分布变尖锐 → 更确定 |
| T=1 | 原分布 |
| T>1 | 分布变平滑 → 更随机、更多样 |

- **注意**：T 不能为负数；T=0 时等价贪心；过高的 T 产生**胡言乱语**。

### 3.3 Top-k：只在高概率项中采样

只保留概率**最高的 k 个** token，其余概率按 0 处理再归一化。限制候选集，避免低概率噪声 token。

### 3.4 Top-p（核采样 / Nucleus Sampling）

只保留**累积概率达到 p** 的那批最小集合（从高到低加），其余清零。候选集**动态可变**，比固定 Top-k 更自适应。

```python
import torch, torch.nn.functional as F

logits = torch.tensor([1.2, 0.5, 0.1, -1.0, -2.0])

def top_p_sample(logits, p=0.9, temp=1.0):
    logits = logits / temp
    probs = F.softmax(logits, dim=-1)
    sorted_p, idx = probs.sort(descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    mask = cum - sorted_p > p          # 去掉超出累计 p 的低概率项
    sorted_p[mask] = 0
    probs = sorted_p / sorted_p.sum()  # 重新归一化
    # 从保留下来的 token 中采样
    return torch.multinomial(probs, 1)

print(top_p_sample(logits))            # 从保留的高概率 token 中随机采样
```

## 四、策略选型对照

| 参数 | 作用 | 典型值 |
|------|------|--------|
| `temperature` | 整体随机性 | 0.1~0.8（创作）/ 0（代码/严谨） |
| `top_k` | 限定前 k 个候选 | 20~100 |
| `top_p` | 限定累积概率 p 的候选 | 0.9~0.95 |
| `max_tokens` | 输出 token 上限 | 按任务 |

> 💡 常用组合：开 top-p 或温度（二选一更主流，避免同时压太狠）；代码/数学用低温，创意写作用中温。

**解码改进**：Beam Search 保留多条路径找更高整体概率，适合翻译/摘要（确定性任务）；但生成长文仍可能重复 → 引入**重复惩罚（repetition penalty）**。

## 五、边界与坑

- ❌ 长上下文推理不做 KV-Cache → 序列越长越慢（O(n³)），甚至显存爆。✅ 标准推理框架都带 KV-Cache。
- ❌ **KV-Cache 用错 batch**（静态 batch 预分配 vs 动态 cache）会 mismatch 报错。✅ 保持 dtype/设备与模型一致。
- ❌ `temperature=0` 还配 `top_p` 采样 → 无意义且可能退化为稳定随机。✅ 低温时直接 greedy 或用 beam。
- ❌ **重复惩罚/采样参数在中文与代码里过猛**会导致语义断裂或标点异常。✅ 按任务微调。
- ❌ 直接采到隐藏结束符/在 agent 场景不配 tool_call 约束 → 生成失控。✅ 工程层加 stop/function schema 约束。

## 六、关联

- 前置知识：Transformer、Self-Attention。
- 同板块：[注意力机制与 Self-Attention](../Seq2Seq-Attention/注意力机制与Self-Attention.md)、[文本预处理与 Tokenizer](../Text-Preprocessing/文本预处理与Tokenizer.md)。
- 进阶：Speculative Decoding、Prefix Caching、结构化解码（structured output）。

## 七、参考

- KV Cache 详解（HuggingFace 博客）— https://huggingface.co/docs/transformers/main/en/llm_opt
- The Curious Case of Neural Text Degeneration (top-p) — https://arxiv.org/abs/1904.09751
- 采样的官方 API（OpenAI）— https://platform.openai.com/docs/api-reference/chat
