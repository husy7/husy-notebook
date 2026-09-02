# -*- coding: utf-8 -*-
"""梯度消失与梯度爆炸 —— 演示代码

覆盖要点：
1. 在深层网络每个 Linear 层挂 backward hook，观测"传到此层的输出梯度范数"；
2. 对比 Sigmoid(饱和激活) 与 ReLU(非饱和激活)+He 初始化在相同深度下的梯度 —— 直观看到消失；
3. 对比初始化方式：Xavier 配合 Sigmoid vs 错误的 He；过大 gain 导致的梯度爆炸；
4. 用 torch.nn.utils.clip_grad_norm_ 做梯度裁剪，演示它如何压住爆炸（只能治爆炸，不能治消失）。

运行（仅 CPU，无需数据文件）：
    python 梯度消失与爆炸_sample.py
"""
import torch
import torch.nn as nn


def build_mlp(depth: int, act: str, hidden: int = 64) -> nn.Sequential:
    """构造: 输入层(8->hidden) + depth 个 hidden 隐层 + 输出层(hidden->1)。

    act: 'sigmoid' | 'tanh' | 'relu'
    """
    acts = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}
    layers = [nn.Linear(8, hidden), acts[act]()]
    for _ in range(depth):
        layers += [nn.Linear(hidden, hidden), acts[act]()]
    layers.append(nn.Linear(hidden, 1))
    return nn.Sequential(*layers)


def init_net(model: nn.Module, kind: str) -> None:
    """按 kind 重设所有 Linear 权重：
    - 'he'     : He(Kaiming) 正态，配合 ReLU 使用
    - 'xavier' : Xavier(Glorot) 均匀，配合 Sigmoid/Tanh 使用
    - 'bad'    : 权重整体放大 gain 倍的"错误初始化"，用于制造爆炸
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if kind == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif kind == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif kind == "bad":
                nn.init.xavier_uniform_(m.weight, gain=2.0)  # gain 过大
            m.bias.data.fill_(0.0)


def grad_norms_per_layer(model: nn.Module, x: torch.Tensor) -> dict:
    """前向 + 反向一次，返回 {seq 索引: 该层输出梯度范数}。

    技巧：register_full_backward_hook 在反向时拿到 (grad_input, grad_output)，
    grad_output[0] 就是"从更深层传回本层输出的梯度"，其范数直接刻画消失/爆炸趋势。
    """
    collected = {}
    handles = []

    def _make(idx: int):
        def _hook(_mod, _gin, gout, _idx=idx):
            collected[_idx] = gout[0].norm().item()
        return _hook

    for i, m in enumerate(model):
        if isinstance(m, nn.Linear):
            handles.append(m.register_full_backward_hook(_make(i)))

    model.zero_grad()
    out = model(x)
    (out.square().mean()).backward()  # 随便一个损失，只为了拿到梯度
    for h in handles:
        h.remove()
    return collected


def total_grad_norm(model: nn.Module) -> torch.Tensor:
    """把所有参数梯度拼成一个向量，算整体 L2 范数（裁剪 API 的度量口径）。"""
    parts = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    if not parts:
        return torch.tensor(0.0)
    return torch.cat(parts).norm()


def demo_theory_floor() -> None:
    """理论下界：Sigmoid 导数最大 0.25，纯连乘时最坏衰减 ~0.25^L。"""
    print("=" * 62)
    print("实验0: 理论视角 —— 每层最大放大系数 0.25，L 层连乘的量级")
    for L in (5, 10, 20, 30):
        print(f"    L={L:>2}: 0.25^L ≈ {0.25 ** L:.3e}   (梯度至少衰减这么多倍)")
    print("    结论: 激活导数 <1 意味着层越深梯度越小, 这是结构性问题,")
    print("          只靠调学习率救不回来——这就是'消失'。\n")


def demo_vanishing() -> None:
    """同深度不同 (激活, 初始化) 组合下，输入侧与输出侧的梯度范数对比。"""
    torch.manual_seed(0)
    x = torch.randn(16, 8)
    depth = 12
    print("=" * 62)
    print(f"实验1: 消失 —— 深度 {depth} 隐层, 激活与初始化不同搭配")
    print("       (输入侧范数 ~ 0 说明梯度到不了浅层 = 消失)")
    print(f"    {'激活':<8}{'初始化':<8}{'输入侧范数':>14}{'输出侧范数':>14}{'衰减倍数':>14}")
    for act, init in (("sigmoid", "xavier"), ("sigmoid", "he"),
                      ("tanh", "xavier"), ("relu", "he"), ("relu", "xavier")):
        net = build_mlp(depth, act)
        init_net(net, init)
        norms = grad_norms_per_layer(net, x)
        first = norms[min(norms)]  # seq 最小 = 最靠近输入的 Linear
        last = norms[max(norms)]   # 最靠近输出的 Linear
        ratio = last / max(first, 1e-300)
        print(f"    {act:<8}{init:<8}{first:>14.3e}{last:>14.3e}{ratio:>14.3e}")

    # 单独展示一条 sigmoid 的"逐层衰减曲线"，看得更清楚
    net = build_mlp(depth, "sigmoid")
    init_net(net, "xavier")
    norms = grad_norms_per_layer(net, x)
    seq = sorted(norms)
    print("\n    示例: sigmoid+xavier 各 Linear 层梯度范数 (从左到右 = 输入→输出)")
    steps = [seq[0], seq[len(seq) // 4], seq[len(seq) // 2],
             seq[3 * len(seq) // 4], seq[-1]]
    print("      " + "  ".join(f"层{i}: {norms[i]:.2e}" for i in steps))
    print("    可见梯度从输出侧回传时被逐层'吃掉', 到输入侧已≈0。\n")


def demo_exploding_and_clip() -> None:
    """过大初始化(gain=2) → 梯度爆炸; clip_grad_norm_ 压回 1.0。"""
    torch.manual_seed(1)
    x = torch.randn(16, 8)
    gain_tried = []
    net = None
    norm0 = torch.tensor(float("inf"))
    # 找一个"爆得明显但数值有限"的配置，避免恰好溢出成 inf 干扰展示
    for gain in (2.0, 1.6, 1.3, 1.1):
        net = build_mlp(depth=10, act="relu")
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                m.bias.data.fill_(0.0)
        loss = net(x).square().mean()
        net.zero_grad()
        loss.backward()
        norm0 = total_grad_norm(net)
        gain_tried.append(gain)
        if torch.isfinite(norm0) and norm0.item() > 1e3:
            break

    print("=" * 62)
    print(f"实验2: 爆炸与梯度裁剪 (尝试 gain={gain_tried}, 采用最后一个)")
    print(f"    裁剪前总梯度范数 = {norm0.item():.4e}")
    clipped_norm = nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    norm1 = total_grad_norm(net)
    print(f"    clip_grad_norm_(max_norm=1.0) 返回(裁剪前范数) = {clipped_norm.item():.4e}")
    print(f"    裁剪后总梯度范数 = {norm1.item():.4e}  (<= max_norm)")
    print("    注: 梯度裁剪=保险丝, 防止一步更新过大/NaN; 它压不住消失。\n")


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}  (CPU 即可运行)\n")
    demo_theory_floor()
    demo_vanishing()
    demo_exploding_and_clip()
    print("观察结论:")
    print("  1) sigmoid/tanh 的网络梯度到不了浅层 → 需要 Xavier + 更浅/残差/BN;")
    print("  2) relu+He 能让梯度平稳传到输入侧; relu+xavier 也会逐层缩水;")
    print("  3) 初始化过大 → 爆炸, clip_grad_norm_ 可兜底。")
