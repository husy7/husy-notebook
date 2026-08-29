# -*- coding: utf-8 -*-
"""
优化器对比：SGD / Momentum / Adam / AdamW —— 典型代码演示
========================================================
覆盖知识点：
  1. 四种优化器在 PyTorch 中的标准用法
  2. 对比实验：在相同网络与数据上，看各优化器的收敛曲线
  3. 理解 weight_decay（L2 正则）与 AdamW 解耦权重衰减
  4. 学习率调度（scheduler）搭配

依赖：pip install torch numpy matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

# =====================================================================
# 一、四种优化器的标准创建方式
# =====================================================================
def make_net():
    """构造一个的小型 MLP 分类器。"""
    return nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

# 每个优化器只需传入 model.parameters() 和学习率
sgd_opt    = optim.SGD(make_net().parameters(), lr=0.05)
mom_opt    = optim.SGD(make_net().parameters(), lr=0.05, momentum=0.9)
adam_opt   = optim.Adam(make_net().parameters(), lr=0.01)
adamw_opt  = optim.AdamW(make_net().parameters(), lr=0.01, weight_decay=1e-2)

# =====================================================================
# 二、对比实验：谁收敛更快？
# =====================================================================
# 制造一个异或（XOR）二分类问题——线性不可分，正好需要非线性网络
X = torch.randn(500, 2)
y = torch.logical_xor(X[:, 0] > 0, X[:, 1] > 0).float().unsqueeze(1)

def train_with_opt(opt_cls, lr, momentum=None, weight_decay=0.0, steps=300,
                   net=None):
    """在 XOR 数据集上训练固定步数，返回每步的损失序列与测试准确率。"""
    net = net or make_net()
    if opt_cls is optim.SGD and momentum:
        opt = opt_cls(net.parameters(), lr=lr, momentum=momentum,
                      weight_decay=weight_decay)
    else:
        opt = opt_cls(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()   # 数值更稳定的二分类损失
    losses = []
    for step in range(steps):
        opt.zero_grad()                # ① 清梯度
        out = net(X)                   # ② 前向
        loss = loss_fn(out, y)         # ③ 损失
        loss.backward()                # ④ 反向
        opt.step()                     # ⑤ 更新
        losses.append(loss.item())
    acc = ((torch.sigmoid(net(X)) > 0.5).float() == y).float().mean().item()
    return losses, acc

# 对比同一算法
results = {}
for name, (cls, lr, kw) in {
    "SGD":        (optim.SGD,   0.05, {"momentum": 0}),
    "Momentum":   (optim.SGD,   0.03, {"momentum": 0.9}),
    "Adam":       (optim.Adam,  0.01, {}),
    "AdamW":      (optim.AdamW, 0.01, {}),
}.items():
    losses, acc = train_with_opt(cls, lr, momentum=kw.get("momentum", 0))
    results[name] = losses
    print(f"{name:9s}: 最终loss={losses[-1]:.3f}  准确率={acc:.2f}")

print("\n→ 通常 Adam/AdamW 收敛最快，SGD 若学习率合适也稳（适合调优后更佳泛化）")

# =====================================================================
# 三、weight_decay：L2 正则化防止过拟合
# =====================================================================
# 用大量参数、少量样本制造过拟合条件
torch.manual_seed(1)
Xw = torch.randn(80, 100)                 # 80 样本, 100 特征（易过拟合）
yw = torch.randint(0, 2, (80, 1)).float()

def train_reg(weight_decay):
    net, opt = make_big_net(), optim.AdamW(make_big_net().parameters(),
                                           lr=1e-2, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn(net(Xw), yw)
        loss.backward(); opt.step()
    return net

def make_big_net():
    return nn.Sequential(nn.Linear(100, 200), nn.ReLU(),
                         nn.Linear(200, 200), nn.ReLU(),
                         nn.Linear(200, 1))

# 权重范数大小对比（weight_decay 越大，参数越小）
for wd in [0.0, 0.1]:
    net = train_reg(wd)
    w_norm = sum(p.norm().item() for p in net.parameters() if p.ndim >= 2)
    print(f"weight_decay={wd}: 权重L2范数之和 = {w_norm:.1f}（越大越易过拟合）")

# =====================================================================
# 四、AdamW 与 (Adam + weight_decay) 的区别：解耦正则
# =====================================================================
# 用代码层面理解：
#   普通 Adam 把 weight_decay 当作 L2 用到"梯度"里 → 会被自适应缩放破坏
#   AdamW 把 weight_decay 直接减到参数上 → 正则稳定（当代 LLM 标配）
net1 = make_net(); opt1 = optim.Adam(net1.parameters(), lr=0.01,
                                     weight_decay=0.01)   # 耦合（默认 L2）
net2 = make_net(); opt2 = optim.AdamW(net2.parameters(), lr=0.01,
                                      weight_decay=0.01)  # 解耦
# 不同之处在于优化器内部的参数更新规则，PyTorch 已在实现层面区分；
# 代码上只需按需选择 Adam 或 AdamW 即可。

# =====================================================================
# 五、学习率调度（warmup + 余弦退火），LLM 训练的标配配方
# =====================================================================
net3 = make_net()
opt3 = optim.AdamW(net3.parameters(), lr=3e-4, weight_decay=0.01)
# 用一个分段预热 + 余弦退火调度器
from torch.optim.lr_scheduler import LambdaLR
warmup_steps, total_steps = 5, 50
def lr_lambda(step):
    if step < warmup_steps:                       # 预热阶段：线性从 0 升到 1
        return step / warmup_steps
    # 余弦退火：从 1 平滑降到 0.05
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.05 + 0.95 * 0.5 * (1 + torch.cos(torch.tensor(prog * 3.1416)))

sched = LambdaLR(opt3, lr_lambda)
print("\n[warmup+余弦] 前 10 步学习率:",
      [round(opt3.param_groups[0]['lr'], 5) for _ in range(10)
       if not sched.step() and False] or
      [round(opt3.param_groups[0]['lr'], 5)] + [0]*0)  # 仅为演示调度存在

for step in range(10):
    opt3.zero_grad()
    loss = nn.BCEWithLogitsLoss()(net3(X), y)
    loss.backward(); opt3.step(); sched.step()

# =====================================================================
# 小结
# =====================================================================
# SGD+Momentum：稳、需调 lr    |  Adam：通用、快
# AdamW：Transformker/LLM 标配 + 解耦 weight_decay
# 深度学习标准配方：AdamW + warmup + 余弦退火。
