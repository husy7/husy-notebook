---
title: "LLM 量化与推理加速：从低比特权重到 vLLM"
tags: [LLM量化, 推理加速, vLLM]
date: 2026-08-30
---

# LLM 量化与推理加速：从低比特权重到 vLLM

## 定义

LLM 自回归解码是**显存带宽瓶颈（memory-bandwidth bound）**任务：每生成一个 token，都要把全部权重从显存搬一遍，算力反而用不满。量化（Quantization）就是用更少的比特（16bit→8bit/4bit/FP8）去近似表示权重与激活，使"每 token 搬运的字节数"直接砍到 1/2 甚至 1/4，从而解码更快、显存占用更小——同一张卡能装下更大模型 / 更长上下文 / 更大 batch。

它解决的问题有三类：①解码速度被显存带宽卡死；②单卡装不下大模型；③服务端单位时间服务不了更多请求（后一半靠推理引擎调度）。核心特征是"位数 / 粒度 / 方法"三者的权衡：位数越低越省却越伤精度；粒度越细（scale 共享范围越小）误差越小但反量化开销越大；方法（RTN/GPTQ/AWQ/GGUF/NF4/FP8）决定量化噪声落在哪里、是否需要校准或训练。代价是精度损失，工程上就是围绕这个三角做取舍。

适用范畴：部署侧的**后训练量化 PTQ**（GPTQ/AWQ/GGUF/NF4，离线一次完成，GPTQ/AWQ 需少量校准数据）与训练侧的**量化感知训练 QAT**；还有 H100 起硬件原生的 FP8（训练/推理兼顾）。需要区分但不冲突的三件事：**权重量化**省的是"搬权重"的带宽；**KV-Cache 量化**省的是中间缓存的显存；**推理引擎调度**（vLLM 等）在算法之外提升并发吞吐。三者解决不同瓶颈、可以叠加使用，其中纯权重 INT4 部署是"装得下 + 跑得快"的最常用组合。

## 原理

为什么位宽直接决定解码速度：自回归每生成一个 token 都要遍历一遍全部权重，粗略吞吐模型为 `每秒 token 数 ≈ 显存带宽 ÷ 每 token 需读取的权重大小`。以 7B 模型为例：fp16 权重 ≈ 14GB、int8 ≈ 7GB、int4 ≈ 3.5GB；A100（2TB/s）跑 7B fp16 → 每 token 读 14GB → 理论上限约 140 token/s，换 int4 → 约 560 token/s（实际还有算子/缓存开销，但方向不变）。

量化基本数学——仿射（非对称）量化把浮点 x 映射到整数 q：

```
q = round(x / scale) + zero_point        scale = (x_max - x_min) / (2^b - 1)
反量化: x̂ = (q - zero_point) · scale
```

权重常用**对称量化**（zero_point=0，实现最简单）：`scale = max|x| / (2^(b-1) - 1)`。量化误差只有两个来源：值域之外的数被**截断(clip)**、值域之内的数被**舍入(round)**；均匀量化下误差 ≈ scale/2。所以 scale 的选择（即"谁来定值域、共享范围多大"）就是量化质量的核心战场——这也是个别离群权重能毁掉整张权重表的原因。

三个精度杠杆：

- **位数**：INT8（几乎无损、最通用）、FP8（新硬件原生，训练/推理兼顾）、INT4（省一半以上，必须配合好方法）。
- **粒度**（scale 的共享范围）：

| 粒度 | 含义 | 误差 | 代价 |
|------|------|------|------|
| per-tensor | 整张权重一个 scale | 大 | 最省 |
| per-channel | 每个输出通道一个 scale | 小 | 略增 |
| per-group | 每 128 个权重一组（如 GPTQ/AWQ 常见 group=128） | 更小 | 反量化次数多 |

- **方法**（PTQ 后训练量化 vs QAT 量化感知训练）：
  - **RTN**：直接四舍五入到最近整数——基线做法，4bit 时损失偏大。
  - **GPTQ**：逐层逐列的**二阶补偿**量化。先把某一列舍入，再用最小二乘（基于该层 Hessian 近似）把这次舍入产生的误差分摊到尚未量化的列上，让该层整体输出尽量不变；需一小份校准数据、一次性离线完成，本质是 OBQ（逐列最优）的高效实现。
  - **AWQ**（Activation-aware Weight Quantization）：观察发现权重里约 **1% 的显著通道**（由激活幅度决定、与输入相关）对量化误差贡献最大；做法是给显著通道乘一个**保护缩放 s**，整体量化后再除回 s——不去单独保住那 1%，而是让量化噪声落在不显著通道上；无需反向传播，比 GPTQ 更稳、校准更快。
  - **bitsandbytes NF4**：4bit 归一化浮点 + **双重量化**（把 scale 再量化成 8bit 以省显存）；transformers `load_in_4bit` 一行接入，是 QLoRA 微调 / 单卡跑大模型的默认路径。
  - **GGUF + llama.cpp(ggml)**：开源模型文件格式（权重 + tokenizer + 超参一体）+ CPU/异构推理栈；用 **k-quants**（Q4_K/Q5_K…，按块混合整数与浮点低位）在纯 CPU 上流畅跑本地模型，Ollama 等工具基于此。

激活也要量化吗？权重量化省的是"搬权重"的带宽，KV-Cache 量化省的是中间缓存显存（见 [[LLM推理与KV-Cache]]）；纯权重 INT4 部署 = 最常用的"装得下 + 跑得快"方案。激活量化的难点在于**离群点(outlier)**，activation 建议 per-token 粒度量化。

引擎层加速（算法之外的调度，解决"单位时间服务更多请求"，与量化正交可叠加）：

- **vLLM PagedAttention**：KV-Cache 按操作系统分页思想管理，消除显存碎片与预留浪费，显存利用率与并发 batch 大幅提升；
- **Continuous Batching**：不再等整批结束，流式地"谁生成完谁出队、新请求即入队"，把 GPU 空闲填满 → 吞吐数倍提升；
- **SGLang / TensorRT-LLM**：前缀复用(RadixAttention)、算子融合与图优化；
- **投机解码(Speculative Decoding)**：小模型先草稿若干 token、大模型一次校验——不改变输出分布地提速（Medusa 等多头草稿是其变体）。

## 应用

典型场景与上手路径（场景 → 方案速查）：

- 本地/CPU 跑开源模型 → llama.cpp GGUF（Q4_K/Q5_K）或 Ollama；
- 单卡 GPU 塞最大模型 → transformers `load_in_4bit`(NF4)，或 AutoGPTQ/AWQ 4bit；
- 生产高吞吐服务 → vLLM + INT8/FP8 + Continuous Batching；
- 微调大模型省显存 → QLoRA = NF4 加载 + LoRA（见 [[LLM微调SFT与LoRA]]）；
- 快速验证流程：训练 fp16 → 保存 → 推理端离线量化（GPTQ/AWQ 用校准数据）→ vLLM 装载服务；量化后再叠加引擎调度与投机解码进一步压榨吞吐/延迟。

常见坑（务必逐条核对）：

- ❌ 以为量化必然"几乎无损"：小模型(<1B)、数学/代码/推理类任务对低位量化更敏感，上 4bit 前先跑 PPL 与代表性任务对比。
- ❌ 校准集与线上分布不符（GPTQ/AWQ 都有校准阶段）→ 用下游任务同分布的数据校准。
- ❌ 只看权重 MSE：真正要看的是激活输出/任务指标的误差；个别离群权重会拉大 per-tensor 的 scale → 无脑上 per-tensor 4bit 通常翻车，先 per-channel/group。
- ❌ 忽略激活离群点(outlier)：这是量化难点所在，activation 建议 per-token 量化。
- ❌ 部署链路不一致：训练 fp16 → 保存 → 推理端再量化时，注意与 `load_in_4bit`/GGUF 转换工具的版本与校准设置一致，别"量化了又没完全量化"。
- ✅ 逐项验证：bits=4/8、group=128/64、per-channel、是否量化 KV cache，用同一批测试 prompt 的困惑度 + 任务得分做小矩阵实验再定方案。

下面用 numpy 演示量化误差机制（对称量化、per-tensor vs per-channel、离群权重的影响），与配套代码 `量化与推理加速_sample.py` 同源：

```python
# 对称量化演示：per-tensor vs per-channel 误差、离群权重(outlier)的影响
# （对应 量化与推理加速_sample.py 的核心逻辑，可直接运行）
# 机制回顾：q = round(clip(x/scale))，scale = max|x|/(2^(bits-1) - 1)；
# 误差来源 = 值域外截断(clip) + 值域内舍入(round)，均匀量化下误差 ≈ scale/2
import numpy as np

def symmetric_quantize(x, bits=4, per_channel=False):
    """对称量化（zero_point=0）。per_channel=False -> per-tensor（整张权重一个 scale，
    最省但误差大）；per_channel=True -> per-channel（每个输出通道一个 scale，误差小）。"""
    if not per_channel:
        scale = np.abs(x).max() / (2 ** (bits - 1) - 1)          # 整张权重一个 scale
    else:
        scale = np.abs(x).max(axis=0, keepdims=True) / (2 ** (bits - 1) - 1)
    q = np.clip(np.round(x / scale), -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1)
    return q, scale

def dequantize(q, scale):
    return q * scale          # 反量化 x̂ = (q - zero_point)·scale，对称量化 zero_point=0

def mse(a, b):
    return float(np.mean((a - b) ** 2))

rng = np.random.default_rng(0)
W = rng.normal(0, 1, size=(256, 256))           # 模拟一层线性权重，形状 (out, in)

# —— 演示 1：per-tensor vs per-channel 的误差差异 ——
q_pt, s_pt = symmetric_quantize(W)                              # 4bit，per-tensor
q_pc, s_pc = symmetric_quantize(W, per_channel=True)            # 4bit，per-channel
print("per-tensor  MSE:", round(mse(W, dequantize(q_pt, s_pt)), 6))
print("per-channel MSE:", round(mse(W, dequantize(q_pc, s_pc)), 6))
# 结论：per-channel 误差更小——各输出通道幅值差异大时，共用一个 scale 会牺牲小通道精度

# —— 演示 2：注入一个离群权重后再比较（离群点如何毁掉 per-tensor）——
W2 = W.copy()
W2[10, 42] = 30.0                               # 人为离群值，远超 4bit 可表示的 ±7 范围
q2_pt, s2_pt = symmetric_quantize(W2)
q2_pc, s2_pc = symmetric_quantize(W2, per_channel=True)
print("含离群 per-tensor  MSE:", round(mse(W2, dequantize(q2_pt, s2_pt)), 6))
print("含离群 per-channel MSE:", round(mse(W2, dequantize(q2_pc, s2_pc)), 6))
# 结论：离群点把 per-tensor 的 scale 撑到 ~2，其余正常权重(~±3σ)量化后塌缩到 0/±1，
# 误差爆炸；per-channel 只惩罚离群所在通道，其余通道几乎不受影响。
# —— 这正是工程上 4bit 必须 per-channel/per-group，以及 AWQ 用"保护缩放 s"
#     而非硬截断来引导量化噪声的设计动机。
```

案例详解：上面的演示 2 直观说明"谁定值域谁决定质量"——对称量化的 scale 由 `max|x|` 决定，个别离群值会撑大整张表的 scale，导致正常权重全部挤进最低几个量化档位、精度崩盘。per-channel 把值域决策下放到每个输出通道，隔离了离群通道的影响；per-group（group=128）则进一步细化到块级。AWQ 的做法与"直接砍掉离群值"不同：它按激活统计识别约 1% 的显著通道并乘上保护缩放 s，让误差集中落在不显著通道上，从而在同等位宽下保住任务精度。

---
## 关联
- 前置：[[LLM推理与KV-Cache]]（自回归解码的显存带宽瓶颈、KV-Cache 占用与量化动机，是该篇的阅读前提；其中 KV-Cache 量化与本篇权重量化互补）
- 类似：[[LLM优化]]（区别是它为全栈优化总览、覆盖面更广；本篇只深入"量化机制 + 推理引擎调度"两层）
- 进阶：[[LLM微调SFT与LoRA]]（QLoRA = NF4 量化加载 + LoRA，把本篇的低位权重量化用于省微调显存）

---
## 对比选型
| 方案 | 核心思想 | 最佳场景 |
|------|---------|----------|
| 本文方案：低位权重量化（INT8/INT4/FP8；GPTQ/AWQ/GGUF/NF4） | 用更少比特近似权重，把每 token 搬运的权重字节砍到 1/2~1/4；误差靠"粒度细化 + 校准分布匹配 + 保护缩放(AWQ)/误差分摊(GPTQ)"控制 | 本地/CPU 跑开源模型、单卡塞最大模型、一切"装得下 + 跑得快"的部署 |
| 替代方案：推理引擎调度加速（vLLM PagedAttention + Continuous Batching、SGLang/TensorRT-LLM） | 不改权重：KV-Cache 分页管理消除显存碎片/预留浪费，流式连续批处理填满 GPU 空闲 | 生产高吞吐在线服务；与量化正交，通常叠加使用 |
| 替代方案：投机解码（Speculative Decoding / Medusa） | 小模型低成本草稿若干 token，大模型一次校验，不改变输出分布地减少大模型串行解码步数 | 延迟敏感且有合适小模型可当草稿模型的服务场景 |

---
## 参考
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers（arxiv 2210.17323）](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration（arxiv 2306.00978）](https://arxiv.org/abs/2306.00978)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention / vLLM（arxiv 2309.06180）](https://arxiv.org/abs/2309.06180)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [llama.cpp（GGUF / k-quants）](https://github.com/ggml-org/llama.cpp)
- [Hugging Face transformers: bitsandbytes 4-bit 量化文档](https://huggingface.co/docs/transformers/quantization/bitsandbytes)

---
## 具体案例
- [[量化与推理加速 实战示例]](量化与推理加速_sample.py)：numpy 演示对称量化、per-tensor vs per-channel 误差差异与离群权重的影响（即上文 python 演示的完整版配套代码）
