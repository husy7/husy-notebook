---
title: "DeepSeek V4.1 Flash 的 MoE 架构与 CSA2"
tags: [deepseek-V4.1, llm, MoE, CSA2]
date: 2026-09-11
---

## 1. MoE 模型基础

- **核心思想**：分而治之。用多个“专家”子网络替换传统 Transformer 的单个 FFN，通过路由网络为每个 Token 动态选择少数专家。
- **稀疏激活**：模型总参数量大，但每个 Token 只激活部分专家，从而控制推理计算量。
- **典型结构**：共享专家 + 路由专家。共享专家吸收通用知识，路由专家处理特定领域知识。

## 2. DeepSeek V4.1 Flash 的 MoE 架构

| 特性 | 设计 |
|---|---|
| 总参数量 | 552B 主干 + 196B Engram 条件记忆参数，含视觉编码器约 763B |
| 激活参数 | 输入 Prefill：8B；输出 Decode：16B |
| 网络结构 | 40 层 Transformer = 20 层因果编码器 + 20 层解码器 |
| 专家配置 | 每层 1 个共享专家 + 384 个路由专家，动态选择激活 |
| 上下文窗口 | 原生支持 100 万 Token，支持多模态视觉理解 |

**设计特点**：非对称 Causal-Encoder-Decoder 结构，针对 Agent 场景“读远多于写”的负载优化。Prefill 计算量接近减半。

## 3. 控制稀疏激活专家的参数与方法

### API 层：间接影响
- **`reasoning_effort`**：支持 `low` / `high` / `max` 或 1–100 整数。提高该值会触发更长推理，间接激活更多或更专业专家，用计算换精度。

### 推理引擎层：直接控制
- **激活专家数量**：
  - `--moe-n-expert`（llama.cpp）：强制指定每 Token 激活专家数。
  - `--override-moe-top-k N`（vLLM fork）：强制每层激活恰好 N 个路由专家。
  - `top_k`（FlashInfer/vLLM）：底层路由选择数量。
- **路由温度 `router_temperature` / `gating_softmax_temp`**：
  - 低温：分布尖锐，选择更确定、效率高。
  - 高温：分布平滑，增加探索性，可能提升泛化。
- **负载均衡**：
  - `expert_load_balance`（Motif-3 Beta）：如 `depth_first` 可激活更多专业专家。
  - 负载补偿：动态调整门控输出，防止少数专家过载。

## 4. CSA2（压缩稀疏注意力 2）架构

### 核心突破
- 传统 Transformer 每层独立存储 KV 缓存，冗余大。
- CSA2 让大部分层**复用其他层已计算的 KV 缓存和稀疏索引**，大幅压缩 KV Cache。

### 三种层模式

| 模式 | Main KV | Indexer K | Top-K 索引 | 职责 |
|---|---|---|---|---|
| **Full** | 完整生成 | 完整生成 | 重新计算 | 锚点层，提供可复用基础 |
| **Reindex** | 复用前序层 | 复用前序层 | 重新计算 | 复用 KV，但重新选择关注位置 |
| **Reuse** | 复用前序层 | 复用前序层 | 复用前序层 | 计算量最小，仅保留 Query 和 SWA KV |

### 分层稀疏索引器
- Full 层扫描完整上下文，构建候选池，规模可达 16,384 位置或 2,048 块。
- 后续层在候选池内做 Top-K 选择，索引成本与上下文总长解耦。

### FP4 KV Cache 协同
- 主 KV 缓存采用 FP4（E2M1），每 16 通道配一个 E4M3 缩放因子。
- 全局 KV Cache 降至每 Token 890 字节，约为 V4-Flash 的 1/4，SSD 占用降至 1/8。
- 从 V1 到 V4.1 Flash，每 Token 全局 KV 缓存从 389,120 字节压缩至 890 字节，约 437 倍。
- 部署收益：HBM 需求 1/4，SSD 需求 1/8，空闲时段缓存命中 API 输入价格低至 0.02 元/百万 Token。

### 代价
- 层模式静态预分配，无法根据输入动态调整，灵活性受限。
- 对 Agentic 工作负载是务实权衡。

## 5. 相关论文与链接

- **CSA2 无单独论文**，技术细节公开在 DeepSeek-V4.1-Flash 技术报告中。
- **技术报告**：[DeepSeek_V41_Tech_Report.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)
- **模型主页**：[DeepSeek-V4.1-Flash on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- **相关解读**：
  - AlphaXiv 论文摘要与讨论
  - MarkTechPost 技术分析
  - InfoQ 中文解读
  - vLLM 官方文档
  - 开发者技术博客

## 6. 总体结论

DeepSeek V4.1 Flash 的 MoE 架构体现务实工程思路：不孤立追求参数规模，而是从 Agent 等真实负载出发，通过非对称架构、CSA2 跨层 KV 复用、FP4 量化等手段，在模型容量、推理成本和部署效率之间取得平衡。这可能是下一代大模型架构竞争的重要方向。