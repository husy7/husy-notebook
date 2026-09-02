---
title: "LLM 推理：KV-Cache 与采样策略"
tags: [LLM, 推理, KV-Cache, 采样, temperature, top-k, top-p]
date: 2026-08-29
---

# LLM 推理：KV-Cache 与采样策略

## 定义

LLM（Decoder-only Transformer）的推理本质是**逐 token 自回归生成**：把已有序列输入模型，得到词表大小的"下一个 token 概率分布" $P(\text{token}_t | \text{context})$，从中选出一个 token 拼回序列，再进入下一步，直到遇到结束符或达到 max_tokens。训练关心"学得好不好"，推理则关心"生成得快不快 + 生成得像不像人"。

这个知识点覆盖推理落地中的两大核心问题：

1. **怎么生成得又快又省？** → **KV-Cache**：缓存历史的 Key/Value，避免每一步都重复计算整段前缀的注意力（解决性能与成本问题）。
2. **怎么从"下一个 token 的概率分布"里挑 token 显得自然、高质量？** → **采样策略**（greedy / temperature / top-k / top-p）：把 logits 分布变成可控、多样、不重复的输出（解决质量问题）。

适用范围：自回归架构的推理/部署环节——在线对话、文本补全、代码生成、翻译与摘要、Agent 工具调用等；它属于**推理阶段**的工程与算法选型，与训练阶段的 loss 优化无直接关系。

核心特征：性能上依赖"缓存历史 + 最小化重复计算"，质量上依赖"对概率分布的缩放/截断 + 随机采样"，二者需要配合使用——一句话概括：**KV-Cache 决定生成多快，采样策略决定生成多好**。

## 原理

**自注意力与朴素做法的复杂度**：自注意力中每个位置都要计算 $Q=W_Q h$、$K=W_K h$、$V=W_V h$，注意力输出为 $\text{softmax}(QK^\top / \sqrt{d_k})V$。若生成第 i 个 token 时每次都把**全部历史序列**重新过一遍模型，前面所有 token 的 K/V 都要重算 → 生成 n 个 token 的总复杂度是 O(n³)，序列越长越慢，极不可用。

**KV-Cache 的核心机制**：第 t 步生成时其实只需要新 token 的 Q（用来查询），而 K/V 可以复用**之前所有步骤缓存的 K/V** 做拼接：

```text
第一步: 算 K1,V1 → 存下来
第二步: 算 K2,V2 → 用 [K1,K2]/[V1,V2] 做注意力 → 存 [K1,K2,V1,V2]
...
每生成一步：
   新 token → 计算新 K,V → 追加进 cache
   attention 用 cache 的全部 K,V 与当前 Q
```

这样每步只算新 token 的 K/V，注意力矩阵乘也从"整段前缀"降为"前缀长度 + 1"的增量计算，总复杂度由 O(n³) 降为 **O(n²)**，大幅省去重复计算。**代价**：KV-Cache 随序列长度线性增长、常驻显存（形状近似 batch × heads × seq_len × head_dim），长上下文 + 大 batch 时它本身就是显存瓶颈。

**显存优化方向**：① KV 量化与缓存压缩（8-bit/4-bit cache）；② MQA/GQA（多查询/分组查询注意力）——让多头**共享 K/V**，显著省显存，是当代 LLM 标配；③ MLA（DeepSeek）把 K/V 做低秩压缩进一步省缓存；④ 长上下文用滑动窗口 + 全局 token 组合。

**采样策略原理**：模型对每一步输出 logits，经 softmax 得到词表上的概率分布。选 token 的几种方式：

- **Greedy（贪心）**：直接选概率最大的 token，确定性最强但易**重复、机械**、缺创造性。
- **Temperature（温度）**：先把 logits 除以温度 T 再做 softmax，公式 $P \propto \exp(\text{logit} / T)$：

| T | 效果 |
|----|------|
| T→0 | 分布变尖锐 → 更确定 |
| T=1 | 原分布 |
| T>1 | 分布变平滑 → 更随机、更多样 |

- **Top-k**：只保留概率**最高的 k 个** token，其余按 0 处理再归一化，限制候选集、滤掉低概率噪声 token。
- **Top-p（核采样 / Nucleus Sampling）**：从高到低累加概率，只保留**累积概率刚好达到 p** 的那批最小集合，其余清零再归一化；候选集**动态可变**，比固定 k 的 top-k 更自适应（论文：The Curious Case of Neural Text Degeneration）。

**解码改进**：Beam Search 同时保留多条候选路径、搜索整句整体概率更高的解，适合翻译/摘要等确定性任务；但长文生成仍易重复 → 引入**重复惩罚（repetition penalty）**。采样不是唯一终点，工程上还可叠加受约束解码（见"应用"的坑）。

## 应用

**典型场景**：在线推理服务（对话/续写/代码补全）、本地部署与显存预算评估、采样参数调优（创作 vs 严谨任务）、翻译摘要等受限生成、Agent 工具调用。

**快速上手步骤**：

1. 优先选用自带 KV-Cache 的标准推理框架（transformers 的 `generate` 内部即开启，vLLM 等推理引擎自动管理 cache），不要手写"每步全量重算"。
2. 代码、数学、抽取等**严谨/确定性任务**：`temperature=0`（等价 greedy）或直接 beam search。
3. **创意写作**：中温 `temperature≈0.7~0.9`，并二选一配合 `top_p≈0.9~0.95` 或 `top_k=20~100`。
4. 设 `max_tokens` 输出上限，按任务控制成本与延迟。
5. 保证 KV-Cache 与模型的 dtype/device 一致，避免静态 batch 预分配与动态 cache 不匹配。

**参数速查表**：

| 参数 | 作用 | 典型值 |
|------|------|--------|
| `temperature` | 整体随机性 | 0.1~0.8（创作）/ 0（代码/严谨） |
| `top_k` | 限定前 k 个候选 | 20~100 |
| `top_p` | 限定累积概率 p 的候选 | 0.9~0.95 |
| `max_tokens` | 输出 token 上限 | 按任务 |

> 💡 常用组合：top-p 或温度**二选一为主**，避免同时压太狠；代码/数学用低温，创意写作用中温。

**常见坑**：

- ❌ 长上下文推理不做 KV-Cache → 序列越长越慢（O(n³)），甚至显存爆。✅ 标准推理框架都带 KV-Cache。
- ❌ KV-Cache 用错 batch（静态 batch 预分配 vs 动态 cache）会 mismatch 报错。✅ 保持 dtype/设备与模型一致。
- ❌ `temperature=0` 还配 `top_p` 采样 → 无意义且可能退化为不稳定随机。✅ 低温时直接 greedy 或用 beam。
- ❌ 重复惩罚/采样参数在中文与代码里过猛 → 语义断裂或标点异常。✅ 按任务微调。
- ❌ 直接采到隐藏结束符；agent 场景不配 tool_call 约束 → 生成失控。✅ 工程层加 stop / function schema 约束（结构化输出）。

```python
import torch, torch.nn.functional as F

# 以 5 个 token 的 logits 为例（值越大表示被选概率越高）
logits = torch.tensor([1.2, 0.5, 0.1, -1.0, -2.0])

def top_p_sample(logits, p=0.9, temp=1.0):
    # 1) 温度缩放：T>1 更随机、T<1 更确定（T→0 近似 greedy）
    logits = logits / temp
    # 2) logits → 概率分布
    probs = F.softmax(logits, dim=-1)
    # 3) 按概率从高到低排序，并算累积概率
    sorted_p, idx = probs.sort(descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    # 4) 核采样核心：去掉超出累计 p 的低概率项（尾部噪声）
    mask = cum - sorted_p > p          # 去掉超出累计 p 的低概率项
    sorted_p[mask] = 0
    probs = sorted_p / sorted_p.sum()  # 对幸存候选重新归一化
    # 5) 从保留下来的 token 中随机采样 1 个
    return torch.multinomial(probs, 1)

print(top_p_sample(logits))            # 每次运行结果可能不同：从幸存的高概率 token 中随机采样
```

**案例详解**：设 `temp=1.0, p=0.9`。softmax 后 5 个 token 的概率约为 [0.505, 0.251, 0.168, 0.056, 0.021]；按从高到低累加得累积概率 0.505 → 0.756 → 0.924 → 0.980 → 1.000。`cum - sorted_p > p` 判断"把当前项之前的部分加起来是否已超过 0.9"，于是最后两个低概率 token（0.056、0.021）被清零；幸存 top-3（合计 0.924）重新归一化后交给 `torch.multinomial` 随机采样。效果：既锚定"最可能"的高质量候选，又保留尾部概率质量带来的多样性，避免只挑最高概率导致重复/机械，也避免采到噪声 token。工程上 top-p 常与 temperature 配合（创作 `temp≈0.7 + top_p≈0.9`；代码/数学直接 `temp=0` 走 greedy）。

---
## 关联
- 前置：[[Transformer]]、[[注意力机制与Self-Attention]]、[[文本预处理与Tokenizer]]
- 类似：[[Beam Search]]（区别是：Beam Search 是确定性搜索——同时保留 top-B 条部分序列、追求整句更高概率，适合翻译/摘要等受限任务，长文易重复需配重复惩罚；本文的采样策略是随机解码——按概率分布抽样，适合开放式创作，但单条采样路径不代表全局最优）
- 进阶：[[Speculative Decoding]]、[[Prefix Caching]]、[[GQA 分组查询注意力]]、结构化解码（structured output / function schema）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：KV-Cache + 概率采样（greedy/temperature/top-k/top-p） | 缓存历史 K/V 免重算前缀（O(n³)→O(n²)）；对 logits 做温度缩放/候选截断后采样选 token | 自回归在线生成：对话、续写、代码补全，兼顾速度与多样性 |
| 朴素全量重算（无 KV-Cache） | 每生成一步都把整段历史重新过一遍注意力 | 仅教学演示/极短序列；长序列 O(n³) 不可用 |
| Beam Search 解码 | 保留 top-B 条部分序列做搜索，最大化整句概率 | 翻译、摘要等确定性受限生成；长文本需配重复惩罚 |
| 结构化/受约束解码 | 在采样中施加 JSON schema / grammar / tool schema 约束 | Agent tool_call、代码生成、需保证格式合法的输出 |

---
## 参考
- [KV Cache 详解（HuggingFace：LLM 推理优化）](https://huggingface.co/docs/transformers/main/en/llm_opt)
- [The Curious Case of Neural Text Degeneration（top-p / Nucleus Sampling，arXiv:1904.09751）](https://arxiv.org/abs/1904.09751)
- [OpenAI Chat Completions API 官方文档（采样参数）](https://platform.openai.com/docs/api-reference/chat)

---
## 具体案例
- [[LLM推理与KV-Cache 实战示例]](LLM推理与KV-Cache_sample.py)
