# -*- coding: utf-8 -*-
"""
PyTorch 训练循环模板 —— 典型代码演示
====================================
覆盖知识点：
  1. Dataset / DataLoader：数据封装、批处理、打乱
  2. 标准训练循环：前向 → 损失 → backward → optimizer.step
  3. 训练/验证模式的切换（train/eval）与 no_grad
  4. 分类模型从数据到评估的完整可运行示例

依赖：pip install torch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

torch.manual_seed(0)

# =====================================================================
# 一、构造一个可分类的数据集（模拟 2 类）
# =====================================================================
n = 1000
# 生成 8 维特征
X = torch.randn(n, 8)
# 规则：第 0 和第 1 维之和的正负决定类别 → 线性可分，方便演示分类
y = (X[:, 0] + X[:, 1] > 0).long()      # 0/1 标签，注意是 long 类型

# =====================================================================
# 二、自定义 Dataset（理解如何封装自己的数据）
# =====================================================================
class CustomDataset(Dataset):
    """自定义数据集：必须实现 __len__ 和 __getitem__。"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)          # 返回样本总数

    def __getitem__(self, idx):
        # 返回第 idx 个 (输入, 标签)；真实项目里这里可做数据增强
        return self.features[idx], self.labels[idx]

# 划分训练/验证
split = int(n * 0.8)
train_ds = CustomDataset(X[:split], y[:split])
val_ds   = CustomDataset(X[split:], y[split:])

# DataLoader：批量 + 打乱 + 多进程取数
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                          num_workers=0)      # Windows 下 num_workers 建议 0
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

# 打印一个 batch 的形状，验证 DataLoader 工作正常
for (xb, yb) in train_loader:
    print("一个 batch:", xb.shape, yb.shape)  # (64,8) (64,)
    break

# =====================================================================
# 三、定义模型、损失、优化器
# =====================================================================
model = nn.Sequential(
    nn.Linear(8, 16),       # 输入 8 维 → 隐层 16
    nn.ReLU(),
    nn.Linear(16, 2),       # 输出 2 个 logits（2 类）
)
loss_fn = nn.CrossEntropyLoss()          # 多分类交叉熵（内部含 softmax）
optimizer = optim.Adam(model.parameters(), lr=1e-2)


# =====================================================================
# 四、训练一个 epoch 的函数
# =====================================================================
def train_one_epoch(loader):
    """在给定 DataLoader 上跑一遍完整的训练。"""
    model.train()                       # 关键：切到训练模式（启用 BN/Dropout）
    total_loss, correct, seen = 0.0, 0, 0
    for inputs, labels in loader:
        optimizer.zero_grad()           # ① 清空上次梯度（否则会累加）
        outputs = model(inputs)         # ② 前向：得到 logits
        loss = loss_fn(outputs, labels) # ③ 计算损失
        loss.backward()                 # ④ 反向传播
        optimizer.step()                # ⑤ 更新参数
        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen   # 平均损失、准确率


# =====================================================================
# 五、验证/评估函数
# =====================================================================
def evaluate(loader):
    """在验证集上评估，不更新参数。"""
    model.eval()                        # 关键：切到评估模式（关 Dropout/BN 用测试统计）
    correct, seen = 0, 0
    with torch.no_grad():               # 不建计算图，省显存、更快
        for inputs, labels in loader:
            outputs = model(inputs)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            seen += len(labels)
    return correct / seen


# =====================================================================
# 六、主循环：多轮 epoch，打印训练与验证表现
# =====================================================================
print("\n===== 训练开始 =====")
for epoch in range(20):
    train_loss, train_acc = train_one_epoch(train_loader)
    val_acc = evaluate(val_loader)
    if epoch % 5 == 0 or epoch == 19:
        print(f"epoch {epoch:2d}: train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

print("\n训练完成。若训练集准确率偏高、验证集明显偏低 → 过拟合；")
print("若两者都低 → 欠拟合或学习率/网络问题。")

# =====================================================================
# 七、完整封装建议：把常用逻辑抽成函数复用
# =====================================================================
def fit(model, train_loader, val_loader, epochs, loss_fn, optimizer):
    """把训练+验证完整封装，返回每轮指标。"""
    history = []
    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(train_loader)
        va_acc = evaluate(val_loader)
        history.append((tr_loss, tr_acc, va_acc))
    return history

# =====================================================================
# 小结与"为什么"速查
# =====================================================================
# model.train()/eval()   → 切换 BN/Dropout 行为，评估不依赖随机丢弃
# optimizer.zero_grad()  → 梯度默认累加，不清会算错
# loss.mean()            → 归一化损失，不受 batch 大小影响
# with torch.no_grad()   → 评估时不建计算图，省显存
# 标签必须 .long()       → CrossEntropy 期望整数类别索引
