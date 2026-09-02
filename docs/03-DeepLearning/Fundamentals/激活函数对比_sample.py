# -*- coding: utf-8 -*-
"""激活函数对比 —— 演示代码

覆盖要点：
1. 在 [-6, 6] 网格上计算各激活的值域与导数，量化"饱和区/平坦区"占比；
2. 直观演示 ReLU 的死亡问题（负半轴导数恒 0）与 LeakyReLU 的缓解；
3. 对比 Sigmoid/Tanh/ReLU/LeakyReLU/GELU 的导数量级 —— 谁会把梯度压小；
4. 验证 GELU 两种实现（erf 精确版 vs tanh 近似版）差异极小。

运行（仅 CPU，不依赖 matplotlib）：
    python 激活函数对比_sample.py
"""
import torch
import torch.nn.functional as F


def sigmoid_deriv(z: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(z)
    return s * (1 - s)          # 最大 0.25


def tanh_deriv(z: torch.Tensor) -> torch.Tensor:
    t = torch.tanh(z)
    return 1 - t * t            # 最大 1.0


def relu_deriv(z: torch.Tensor) -> torch.Tensor:
    return (z > 0).to(z.dtype)  # 负半轴恒 0


def leaky_deriv(z: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, alpha))


def gelu_deriv(z: torch.Tensor) -> torch.Tensor:
    """GELU = x * Phi(x)，Phi 为标准正态 CDF。用精确 erf 版求导。"""
    cdf = 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0))))
    pdf = torch.exp(-0.5 * z * z) / torch.sqrt(torch.tensor(2.0 * torch.pi))
    return cdf + z * pdf


def stats(z: torch.Tensor, fn, d_fn):
    """返回 (输出范围, 导数范围, 平坦区占比[|导数|<0.02])。"""
    y = fn(z)
    d = d_fn(z)
    flat = (d.abs() < 0.02).float().mean().item()
    return (y.min().item(), y.max().item()), (d.min().item(), d.max().item()), flat


def demo_value_and_deriv() -> None:
    """核心表：值域 + 导数量级 + 平坦(饱和/死亡)区占比。"""
    z = torch.linspace(-6.0, 6.0, 1201)  # 覆盖饱和区与线性区
    cases = [
        ("Sigmoid",    torch.sigmoid,      sigmoid_deriv),
        ("Tanh",       torch.tanh,         tanh_deriv),
        ("ReLU",       lambda t: F.relu(t), relu_deriv),
        ("LeakyReLU",  lambda t: F.leaky_relu(t, 0.01), leaky_deriv),
        ("GELU",       lambda t: F.gelu(t), gelu_deriv),
    ]
    print("=" * 74)
    print("实验1: [-6,6] 网格上的值域 / 导数 / 平坦区占比(|f'|<0.02)")
    print(f"    {'激活':<10}{'输出范围':>22}{'导数范围':>20}{'平坦占比':>10}")
    for name, fn, dfn in cases:
        (ylo, yhi), (dlo, dhi), flat = stats(z, fn, dfn)
        print(f"    {name:<10}({ylo:>6.2f},{yhi:>6.2f})  "
              f"({dlo:>6.2f},{dhi:>6.2f})  {flat * 100:>8.1f}%")

    print("\n    解读:")
    print("      - Sigmoid/Tanh 在两端导数≈0(饱和区占比高) → 深网里梯度被逐层压没;")
    print("      - Tanh 输出零中心, 但饱和问题与 Sigmoid 同类;")
    print("      - ReLU 正半轴导数恒1、负半轴恒0(平坦区=死亡区, 约一半);")
    print("      - LeakyReLU 的平坦区全是 α=0.01, 梯度不绝; GELU 平滑无硬零区。\n")


def demo_dead_relu() -> None:
    """死亡 ReLU：一个权重初始化/偏置不当的神经元可能对全部样本输出 0。"""
    torch.manual_seed(0)
    z = torch.randn(10000) - 2.0          # 输入整体偏负（模拟权重偏移/大负偏置）
    r = F.relu(z)
    dead_ratio = (r == 0).float().mean().item()
    print("=" * 74)
    print(f"实验2: 死亡神经元演示 —— 输入 N(-2,1), 10000 个样本")
    print(f"    ReLU      输出恒0的占比: {dead_ratio * 100:6.1f}%  "
          f"(该分支梯度也恒0, 无法自愈)")
    lr_out = F.leaky_relu(z, 0.01)
    lr_grad = torch.where(z > 0, torch.ones_like(z), torch.full_like(z, 0.01))
    print(f"    LeakyReLU 该区间的梯度  : {lr_grad[z <= 0].min().item():.3f} "
          f"(仍有小梯度, 可缓慢恢复)")
    print(f"    对比: ReLU 在负区间的梯度 = {relu_deriv(z[z <= 0]).max().item()}  "
          f"-> 完全断流\n")


def demo_gelu_approx() -> None:
    """GELU: 精确(erf) vs tanh 近似, 差异应 ~1e-3 量级。"""
    z = torch.linspace(-8.0, 8.0, 2001)
    exact = F.gelu(z, approximate="none")     # 默认, 用 erf
    approx = F.gelu(z, approximate="tanh")    # 快速近似
    print("=" * 74)
    print("实验3: GELU 两种实现对比")
    print(f"    max|erf精确版 - tanh近似版| = {(exact - approx).abs().max().item():.2e}")
    print("    -> 训练/加载预训练模型时务必与原作者选同一版本。\n")


def demo_derivative_scale_table() -> None:
    """导数峰值表：直观回答'激活会把梯度放大到多少'。"""
    print("=" * 74)
    print("实验4: 导数峰值 —— 反向传播单层最多能乘的放大系数")
    z = torch.linspace(-2.0, 2.0, 400001)
    gmax = gelu_deriv(z).max().item()
    rows = [
        ("Sigmoid", 0.25,   "负半轴也≤0.25"),
        ("Tanh",    1.0,    "峰值在原点, 两端饱和"),
        ("ReLU",    1.0,    "负半轴=0(死亡区)"),
        ("LeakyReLU", 1.0,  "负半轴=α(0.01)"),
        ("GELU",    gmax,   "峰值≈1.13, 平滑无硬零区"),
    ]
    print(f"    {'激活':<10}{'导数峰值':>10}    {'备注':<20}")
    for name, val, note in rows:
        print(f"    {name:<10}{val:>10.3f}    {note}")
    print("\n    结论: 一层0.25无所谓, 十几层0.25连乘=消失;\n"
          "          让每层放大系数≈1(ReLU+好初始化/归一化) 梯度才能走远。")


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}\n")
    demo_value_and_deriv()
    demo_dead_relu()
    demo_gelu_approx()
    demo_derivative_scale_table()
