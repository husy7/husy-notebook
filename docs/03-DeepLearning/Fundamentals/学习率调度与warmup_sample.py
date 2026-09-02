# -*- coding: utf-8 -*-
"""学习率调度与 Warmup —— 演示代码

覆盖要点：
1. 常用调度器语义：StepLR / MultiStepLR / CosineAnnealingLR / ExponentialLR；
2. Warmup 的两种实现：LambdaLR 手写斜坡 与 LinearLR+SequentialLR 串联；
3. 通用配方：linear warmup + cosine decay（大模型/Transformer 训练惯例）；
4. 调度节奏：epoch 级（每 epoch step 一次）vs batch 级（每 batch step 一次）；
5. ReduceLROnPlateau：由验证指标驱动，指标停滞才降 lr。

运行（仅 CPU，用假损失模拟训练进度）：
    python 学习率调度与warmup_sample.py
"""
import math

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ExponentialLR, LambdaLR, LinearLR,
    MultiStepLR, ReduceLROnPlateau, SequentialLR, StepLR,
)


def make_dummy_optimizer(lr: float = 0.1):
    """一个只有占位参数的小优化器，专用于打印调度曲线。"""
    p = torch.randn(2, 2, requires_grad=True)
    return SGD([p], lr=lr)


def run_epoch_level(name, build_sched, epochs, *, metric=None):
    """epoch 级演示：每 epoch 末尾 scheduler.step()（或喂指标给 plateau）。"""
    opt = make_dummy_optimizer(0.1)
    sched = build_sched(opt)
    print(f"\n--- {name} ({epochs} epochs) ---")
    print(f"    {'epoch':>5} {'lr':>10}   {'(伪)loss':>10}")
    fake_loss = 2.0
    for ep in range(epochs):
        opt.zero_grad()
        p = opt.param_groups[0]["params"][0]
        # 模拟优化：往 0 走一点点（仅为了保持 optimizer 状态有意义）
        p.data.mul_(0.999)
        opt.step()
        # 衰减只是"演示数值": 这里让伪 loss 缓慢下降供 plateau 判断
        fake_loss = fake_loss * 0.93 + 0.01
        # ---- epoch 级调度: 每 epoch step 一次 ----
        if isinstance(sched, ReduceLROnPlateau):
            sched.step(fake_loss)          # 需要喂一个验证指标
        else:
            sched.step()
        lr = sched.get_last_lr()[0]
        print(f"    {ep + 1:>5} {lr:>10.6f}   {fake_loss:>10.4f}")
    return sched.get_last_lr()[0]


def run_batch_level(name, build_sched, total_steps):
    """batch 级演示：one-cycle / 按 iteration 衰减的调度器在每 batch 后 step。"""
    opt = make_dummy_optimizer(0.1)
    sched = build_sched(opt)
    print(f"\n--- {name} ({total_steps} steps) ---")
    print(f"    {'step':>6} {'lr':>10}")
    for s in range(total_steps):
        opt.step()          # 每个 batch 一次参数更新
        sched.step()        # 每个 batch 推进一次调度
        if s % max(1, total_steps // 10) == 0 or s == total_steps - 1:
            print(f"    {s + 1:>6} {sched.get_last_lr()[0]:>10.6f}")


def demo_schedulers() -> None:
    """三个 epoch 级常用调度器。"""
    print("=" * 66)
    print("实验1: epoch 级调度器（每 epoch step 一次）")
    run_epoch_level(
        "StepLR(step_size=10, gamma=0.5) —— 每10个epoch砍半",
        lambda opt: StepLR(opt, step_size=10, gamma=0.5), 30)
    run_epoch_level(
        "MultiStepLR(milestones=[10,20], gamma=0.1) —— 里程碑骤降",
        lambda opt: MultiStepLR(opt, milestones=[10, 20], gamma=0.1), 30)
    run_epoch_level(
        "ExponentialLR(gamma=0.9) —— 每个epoch×0.9",
        lambda opt: ExponentialLR(opt, gamma=0.9), 30)
    run_epoch_level(
        "CosineAnnealingLR(T_max=30, eta_min=1e-5) —— 余弦平滑衰减",
        lambda opt: CosineAnnealingLR(opt, T_max=30, eta_min=1e-5), 30)


def demo_warmup_cosine() -> None:
    """通用配方: linear warmup (5 epoch) + cosine decay (25 epoch)。"""
    print("=" * 66)
    print("实验2: linear warmup + cosine decay (warmup=5, 共30 epoch)")
    opt = make_dummy_optimizer(0.1)  # 目标 lr = 0.1

    warmup_epochs, total = 5, 30

    # 写法A: 一个 LambdaLR 覆盖两段（lr = base_lr * scale(epoch)）
    def scale(epoch: int) -> float:
        if epoch < warmup_epochs:                      # 热身: 0.2, 0.4, ..., 1.0
            return (epoch + 1) / warmup_epochs
        t = epoch - warmup_epochs                      # 之后余弦从 1 平滑降到 ~0
        remain = total - warmup_epochs
        return 0.5 * (1.0 + math.cos(math.pi * t / remain))

    sched = LambdaLR(opt, lr_lambda=scale)
    print(f"    {'epoch':>5} {'lr(LambdaLR两段式)':>18}")
    for ep in range(total):
        opt.step()
        sched.step()                                   # epoch 级: 每 epoch 推进一步
        if ep in (0, 1, 4, 5, 10, 20, 29):
            print(f"    {ep + 1:>5} {sched.get_last_lr()[0]:>18.6f}")
    print("    (写法A: LambdaLR 把 lr = base_lr × scale(epoch), 一行函数表达任意策略)")

    # 写法B: SequentialLR(LinearLR warmup + CosineAnnealingLR), 官方串联 API
    opt2 = make_dummy_optimizer(0.1)
    sched = SequentialLR(
        opt2,
        schedulers=[
            LinearLR(opt2, start_factor=0.2, total_iters=warmup_epochs - 1),
            CosineAnnealingLR(opt2, T_max=total - warmup_epochs, eta_min=1e-5),
        ],
        milestones=[warmup_epochs - 1],   # 前 4 个 epoch 用 warmup 段, 之后切 cosine
    )
    print(f"    {'epoch':>5} {'lr(SequentialLR官方串联)':>22}")
    for ep in range(total):
        opt2.step()
        sched.step()
        if ep in (0, 1, 4, 5, 10, 20, 29):
            print(f"    {ep + 1:>5} {sched.get_last_lr()[0]:>22.6f}")


def demo_batch_level() -> None:
    """batch 级调度：OneCycleLR(内含先升后降) 需每个 batch step。"""
    print("=" * 66)
    print("实验3: batch 级调度 —— OneCycleLR(max_lr=0.1, 总步数=100)")
    run_batch_level(
        "OneCycleLR(每batch step)",
        lambda opt: torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=0.1, total_steps=100), 100)


def demo_plateau() -> None:
    """ReduceLROnPlateau：监控验证指标, 停滞 5 轮则 lr×0.5。"""
    print("=" * 66)
    print("实验4: ReduceLROnPlateau(patience=5, factor=0.5)")
    opt = make_dummy_optimizer(0.1)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    print(f"    {'epoch':>5} {'lr':>10}   {'val_loss':>10} {'触发降lr?':>8}")
    val = 1.0
    for ep in range(40):
        opt.step()
        # 模拟: 前20轮缓慢下降, 后20轮平台期 -> 触发 plateau 降 lr
        if ep < 20:
            val = val * 0.94
        else:
            val = val * 0.999 + 0.005   # 几乎不动 = 平台
        old = sched.get_last_lr()[0] if hasattr(sched, "get_last_lr") else opt.param_groups[0]["lr"]
        sched.step(val)
        new = sched.get_last_lr()[0]
        if old != new or ep % 5 == 0:
            print(f"    {ep + 1:>5} {new:>10.6f}   {val:>10.4f}   {'是' if old != new else '-':>8}")


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}")
    demo_schedulers()
    demo_warmup_cosine()
    demo_batch_level()
    demo_plateau()
    print("\n要点回顾:")
    print("  1) epoch级调度每epoch step一次; iteration级(OneCycle)每batch step一次;")
    print("  2) warmup用 LinearLR/LambdaLR 斜坡; 大模型配方= linear warmup + cosine;")
    print("  3) ReduceLROnPlateau 不需要知道衰减时机, 交给验证指标。")
