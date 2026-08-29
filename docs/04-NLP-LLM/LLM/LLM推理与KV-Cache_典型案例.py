# -*- coding: utf-8 -*-
"""
LLM 推理：KV-Cache 与采样策略 —— 典型代码演示
===============================================
覆盖知识点：
  1. 贪心 / Temperature / Top-k / Top-p 采样器的手写实现
  2. KV-Cache 的简单实现与性能对比（关键原理演示）
  3. HuggingFace pipeline 的采样参数实战
  4. 各类参数对生成多样性的影响演示

依赖：pip install torch numpy transformers(可选)
"""

import math
import time
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# =====================================================================
# 一、四种采样策略的实现（从 logits 分布里挑下一个 token）
# =====================================================================
def greedy_decode(logits):
    """贪心：直接选概率最大的 token。确定性、易重复。"""
    return logits.argmax(dim=-1, keepdim=True)


def temperature_sample(logits, temperature=1.0):
    """温度采样：把 logits 除以 T 再 softmax。
       T<1 → 更确定；T=1 → 原分布；T>1 → 更随机。"""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)   # 按概率分布采样一个


def top_k_sample(logits, k=3, temperature=1.0):
    """Top-k：只保留概率最高的 k 个，其余置 0 再归一化。"""
    logits = logits / temperature
    kth = torch.topk(logits, k)[0][..., -1:]          # 第 k 大的值
    logits[logits < kth] = float("-inf")              # 清除非 top-k
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


def top_p_sample(logits, p=0.9, temperature=1.0):
    """Top-p（核采样）：保留累积概率达 p 的最小集合，动态候选数。"""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, idx = probs.sort(descending=True)   # 从高到低排序
    cumsum = torch.cumsum(sorted_probs, dim=-1)       # 累积概率
    # 丢弃那些"加上去就超过 p"的低概率项
    remove = cumsum - sorted_probs > p
    sorted_probs[remove] = 0.0
    # 重新归一化到总和为 1（放回原顺序）
    unsorted = torch.zeros_like(probs).scatter(-1, idx, sorted_probs)
    return torch.multinomial(unsorted / unsorted.sum(dim=-1, keepdim=True), 1)


# 用一个假 logits 分布演示这几种策略的输出差异
logits = torch.tensor([[1.2, 0.8, 0.1, -0.5, -2.0]])   # 词表仅 5 token

print("[采样策略演示] 原始 logits:", logits.tolist())
print("[贪心] 选中 token:", greedy_decode(logits).item())
print("[Temperature=2.0(高)] 随机性大增 → 可能选到低频词")

# 跑多次观察 Top-p 在"动态候选"上的效果
print("\n[top_p=0.9] 连续采样 20 次:", 
      [top_p_sample(logits.clone(), p=0.9).item() for _ in range(20)])
print("[top_k=2]   连续采样 20 次:",
      [top_k_sample(logits.clone(), k=2).item() for _ in range(20)])

# =====================================================================
# 二、KV-Cache 的简单原理演示 + 性能对比
# =====================================================================
def generate_no_cache(proj_k, proj_v, k_list, v_list, new_k, new_v):
    """朴素做法：每次都拿"全部序列"重新算 K/V。返回需要重算的长度。"""
    # 模拟：无 cache 时新 token 到来，必须重新处理整个前缀
    return len(k_list) + len(new_k)   # 需要重算的 K 长度


def generate_with_cache(proj_k, proj_v, k_cache, v_cache, new_k, new_v):
    """KV-Cache 做法：只计算新 token 的 K/V 并追加进缓存复用。"""
    k_cache = torch.cat([k_cache, new_k], dim=1)   # 追加到已有缓存
    v_cache = torch.cat([v_cache, new_v], dim=1)
    # 注意力只需一次拼接后的 cache，不必重算前缀
    return k_cache, v_cache


# 模拟生成 100 个 token，对比"每次全量重算" vs "KV-Cache 复用"
seq_len, d = 10, 8
k_cache = torch.randn(1, seq_len, d)   # 初始已缓存的一部分 K
v_cache = torch.randn(1, seq_len, d)

recompute_units = 0
for _ in range(100):
    new_k = torch.randn(1, 1, d)       # 每步新生成的 token 的 K
    new_v = torch.randn(1, 1, d)
    # 无 cache：需要重算的 token 数 = 前缀 + 新
    recompute_units += generate_no_cache(None, None, k_cache, v_cache, new_k, new_v)
    # 有 cache：只算 1 个新 token
    k_cache, v_cache = generate_with_cache(None, None, k_cache, v_cache, new_k, new_v)

print("\n[KV-Cache 演示]")
print(f"生成 100 个 token：无 cache 需重算 {recompute_units} 次单元")
print(f"                有 cache 只需计算 {100} 个新单元（大幅减少重复计算）")
avg_cache_len = k_cache.shape[1]
print(f"[代价] KV-Cache 大小递增, 最终 K cache 长度 = {avg_cache_len},", 
      "是显存随序列增长的来源")

# =====================================================================
# 三、用 HuggingFace 演示真实采样参数（若有 transformers 环境）
# =====================================================================
try:
    from transformers import pipeline
    gen = pipeline("text-generation", model="distilgpt2")

    prompts = "The best way to learn AI is"
    out = gen(prompts, max_new_tokens=20, do_sample=True,
              temperature=0.7, top_p=0.9, num_return_sequences=1)
    print("\n[HuggingFace] 采样生成:", out[0]["generated_text"])
except Exception as e:
    print("\n[可选] 未配置 transformers 生成环境（", type(e).__name__, "），跳过")

# =====================================================================
# 小结与参数速查
# =====================================================================
# temperature: 全局随机性   top_k: 固定候选数   top_p: 动态候选(核采样)
# 代码/数学用低温+确定性；创作用中高温+采样；KV-Cache 是长上下文推理提速关键。
