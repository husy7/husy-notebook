"""量化的核心机制演示（numpy，可直接运行，无需 GPU/模型下载）

演示内容：
  1) 对称均匀量化/反量化：q = round(x / scale)，scale = max|x|/(2^(b-1)-1)
  2) 粒度对比：per-tensor vs per-channel —— 用含"离群行"的权重矩阵演示
     为什么 per-channel 误差远小于 per-tensor（GPTQ/AWQ 都用 group/channel 粒度）；
  3) 位数对比：fp16 / int8 / int4 的反量化误差与"权重-输出误差"的关系；
  4) 附：显存带宽直觉计算 + 生产工具(GPTQ/AWQ/GGUF/NF4/vLLM)的接入示例(注释)。

配套笔记：量化与推理加速-note.md
"""
import numpy as np

rng = np.random.default_rng(0)


def quantize_symmetric(x: np.ndarray, bits: int, axis=None) -> tuple:
    """对称量化：q = round(x/scale) 截断到 [-Q, Q]，返回 (q, scale) 与反量化值。
    axis=None → per-tensor(全矩阵共享一个 scale)；axis=-1 → per-channel。"""
    qmax = 2 ** (bits - 1) - 1
    if axis is None:
        scale = np.abs(x).max() / qmax
        scale = np.array(scale)
    else:
        scale = np.abs(x).max(axis=axis, keepdims=True) / qmax
        scale = np.where(scale == 0, 1.0, scale)          # 防整行/整列为 0
    q = np.clip(np.round(x / scale), -qmax, qmax)
    x_hat = q * scale                                     # 反量化（对称 zp=0）
    return q.astype(np.int8) if bits <= 8 else q.astype(np.int16), x_hat, scale


def report(tag: str, x: np.ndarray, x_hat: np.ndarray) -> float:
    """相对误差 + 绝对误差 双指标。"""
    rel = np.linalg.norm(x - x_hat) / (np.linalg.norm(x) + 1e-12)
    mae = np.abs(x - x_hat).mean()
    print(f"  {tag:<44} 相对L2误差={rel:.4f}  平均绝对误差={mae:.4f}")
    return rel


def main() -> None:
    # ---- 0) 显存/带宽直觉 ----
    params, bits = 7e9, 16
    print(f"== 显存直觉: 7B 模型 fp16 ≈ {params*bits/8/1e9:.1f}GB, "
          f"int8 ≈ {params*8/8/1e9:.1f}GB, int4 ≈ {params*4/8/1e9:.1f}GB ==")

    # ---- 1) 正常分布的权重: per-tensor 就够好 ----
    print("\n== 1) 高斯权重 W~N(0,1)，per-tensor 量化 ==")
    W = rng.normal(0, 1, (64, 64))
    for b in (8, 4):
        _, x_hat, scale = quantize_symmetric(W, b)
        report(f"bits={b} per-tensor (scale={float(scale):.4f})", W, x_hat)

    # ---- 2) 加入"离群行/通道"（真实 LLM 权重常见）----
    print("\n== 2) 权重矩阵某一行数值特别大（离群）→ 粒度对比 ==")
    W2 = W.copy()
    W2[0, :] = W2[0, :] * 100                 # 制造一行离群权重
    for bits in (8, 4):
        # per-tensor: 离群行把 scale 拉大 → 其余 63 行全被"压缩"到低分辨率
        _, pt, _ = quantize_symmetric(W2, bits)
        # per-channel: 每行自己的 scale → 离群只惩罚自己那一行
        _, pc, _ = quantize_symmetric(W2, bits, axis=-1)
        report(f"bits={bits} per-tensor", W2, pt)
        report(f"bits={bits} per-channel", W2, pc)
        print()

    # ---- 3) 端到端意义: 权重误差如何传导到输出 ----
    print("== 3) 误差传导: 用量化权重算一层线性输出 (y = x·W) ==")
    x = rng.normal(0, 1, (8, 64))
    _, Wq, _ = quantize_symmetric(W2, 4, axis=-1)          # per-channel int4
    _, Wt, _ = quantize_symmetric(W2, 4)                   # per-tensor int4
    y_ref = x @ W2
    rel_ch = np.linalg.norm(x @ Wq - y_ref) / np.linalg.norm(y_ref)
    rel_tt = np.linalg.norm(x @ Wt - y_ref) / np.linalg.norm(y_ref)
    print(f"  per-channel int4 输出相对误差 = {rel_ch:.4f}")
    print(f"  per-tensor  int4 输出相对误差 = {rel_tt:.4f}  ← 离群权重放大到输出")
    print("""
  结论: 这就是 GPTQ/AWQ 采用 per-group/per-channel 的原因——用"细粒度 scale"
        把离群权重的影响局部化；AWQ 更进一步只保护 ~1% 显著通道(乘缩放 s 再量化)。
  注意: 真实 LLM 的难点在"激活离群"(token 维), 权重量化还需要配合校准数据,
        请勿仅凭权重 MSE 下结论。""")

    # ---- 4) 生产接入速览（注释: 需要装对应库与模型）----
    print("""
== 生产接入速览（按需选用，非本脚本执行部分）==
# transformers + bitsandbytes (NF4 双重量化, 4bit 加载, 单卡跑大模型):
#   from transformers import AutoModelForCausalLM, BitsAndBytesConfig
#   model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
#               quantization_config=BitsAndBytesConfig(load_in_4bit=True))
#
# GPTQ/AWQ 预量化模型 + vLLM 服务 (高吞吐):
#   pip install vllm
#   vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --max-model-len 8192
#
# CPU/本地: llama.cpp GGUF (Q4_K_M) 或 Ollama:
#   ollama run qwen2.5:7b
#
# 度量陷阱: 4bit 前后对比 PPL + 下游任务分数, group=128 为常用默认值。
""")


if __name__ == "__main__":
    main()
