"""RNN 与 LSTM 核心机制实验（PyTorch，可直接运行，CPU 即可）

三部分：
  Part 1  手写 RNN 前向 (h_t = tanh(W_ih·x_t + b_ih + W_hh·h_{t-1} + b_hh))
          与 nn.RNN 输出对齐 → 验证"共享权重 + 时间递归"就是它的全部；
  Part 2  手写 LSTM 门控前向（i/f/g/o 顺序与 PyTorch 一致）与 nn.LSTM 对齐
          → 看清三条门 + 细胞状态 c 的加性更新；
  Part 3  梯度实验：同样的序列任务上对比 RNN 与 LSTM 对"早期输入"的梯度量级，
          直观演示梯度消失问题与 LSTM 的缓解。

配套笔记：RNN与LSTM-note.md
"""
import torch
import torch.nn as nn

torch.manual_seed(0)


# ---------------- Part 1: 手写 RNN 前向 ----------------
def manual_rnn_forward(W_ih, b_ih, W_hh, b_hh, x, h0):
    """x: (T, B, in)，h0: (B, hidden)。逐时间步递归，权重全程共享。"""
    T, B, _ = x.shape
    hidden = W_hh.shape[0]
    h = h0
    outs = []
    for t in range(T):
        h = torch.tanh(x[t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh)  # 共享参数
        outs.append(h)
    return torch.stack(outs), h


def check_rnn():
    print("== Part 1: 手写 RNN vs nn.RNN ==")
    T, B, IN, H = 6, 3, 5, 8
    cell = nn.RNN(IN, H, batch_first=False)      # 默认 tanh
    x = torch.randn(T, B, IN)
    h0 = torch.randn(B, H)
    with torch.no_grad():
        h_manual, hn_manual = manual_rnn_forward(
            cell.weight_ih_l0, cell.bias_ih_l0, cell.weight_hh_l0, cell.bias_hh_l0, x, h0)
        h_torch, hn_torch = cell(x, h0.unsqueeze(0))     # 3D 输入要求 hx 带层维 (L,B,H)
    hn_torch = hn_torch[0]                               # 单层 → 取第 0 层
    err = (h_manual - h_torch).abs().max().item()
    print(f"  全序列隐状态最大误差 = {err:.2e}  (应 ~1e-6，即实现等价)")
    return err


# ---------------- Part 2: 手写 LSTM 门控 ----------------
def manual_lstm_forward(lstm: nn.LSTM, x, h0=None, c0=None):
    """按 PyTorch 的权重排布手写单层 LSTM：
    weight_ih_l0 的每 hidden 行为一组，顺序 = [i, f, g, o]。
    公式:
      i = sigmoid(W_i x + b_i + W_hi h + b_hi)  输入门
      f = sigmoid(W_f x + b_f + W_hf h + b_hf)  遗忘门
      g = tanh(W_g x + b_g + W_hg h + b_hg)     候选记忆
      o = sigmoid(W_o x + b_o + W_ho h + b_ho)  输出门
      c = f*c_prev + i*g     h = o*tanh(c)
    """
    T, B, _ = x.shape
    H = lstm.hidden_size
    W_ih, b_ih = lstm.weight_ih_l0, lstm.bias_ih_l0
    W_hh, b_hh = lstm.weight_hh_l0, lstm.bias_hh_l0
    h = h0 if h0 is not None else torch.zeros(B, H)
    c = c0 if c0 is not None else torch.zeros(B, H)
    outs = []
    for t in range(T):
        gates = x[t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh        # (B, 4H)
        i, f, g, o = gates.chunk(4, dim=1)                      # 依序切出四门
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g                                       # 细胞状态：门控加性更新
        h = o * torch.tanh(c)
        outs.append(h)
    return torch.stack(outs), (h, c)


def check_lstm():
    print("\n== Part 2: 手写 LSTM 门控 vs nn.LSTM ==")
    T, B, IN, H = 6, 3, 5, 8
    cell = nn.LSTM(IN, H, batch_first=False)
    x = torch.randn(T, B, IN)
    h0, c0 = torch.randn(B, H), torch.randn(B, H)
    with torch.no_grad():
        h_manual, (hn_m, cn_m) = manual_lstm_forward(cell, x, h0, c0)
        h_torch, (hn_t, cn_t) = cell(x, (h0.unsqueeze(0), c0.unsqueeze(0)))
    hn_torch, cn_torch = hn_t[0], cn_t[0]
    err_h = (h_manual - h_torch).abs().max().item()
    err_c = (cn_m - cn_t).abs().max().item()
    print(f"  h 序列最大误差 = {err_h:.2e}，c 终态最大误差 = {err_c:.2e}")
    return max(err_h, err_c)


# ---------------- Part 3: 长程梯度实验 ----------------
def gradient_probe():
    print("\n== Part 3: 长程梯度量级对比（解释为什么 LSTM 记得住、RNN 记不住）==")
    T, IN, H = 20, 4, 10
    print(f"  {'模型':<10} {'|grad x_0|':>12} {'|grad x_{T-1}|':>14}  梯度比(早/晚)")
    for name, cell in [("RNN(tanh)", nn.RNN(IN, H)), ("LSTM", nn.LSTM(IN, H))]:
        x = torch.randn(T, 1, IN) * 0.5
        x.requires_grad_(True)
        out, _ = cell(x)
        (out[-1] ** 2).sum().backward()           # 损失只依赖最后时刻的输出
        g0 = x.grad[0].norm().item()              # 最早输入 x_0 的梯度范数
        gT = x.grad[-1].norm().item()             # 最近输入 x_{T-1} 的梯度范数
        print(f"  {name:<10} {g0:12.3e} {gT:14.3e}  {g0 / max(gT, 1e-12):.3e}")
        del x, cell

    print("""
  解读: RNN 里 x_0 的梯度要穿过 T-1 次 tanh'·W_hh 连乘 → 指数衰减(可能下溢为 0)；
        LSTM 的细胞状态 c 是"门控加性"通道, 误差沿 c 反传只乘遗忘门,
        远距离梯度仍然可观 → 能学到长程依赖。
  提示: 若 RNN 的 |grad x_0| 打印为 0.000e+00, 说明连乘已下溢——这本身就是
        "梯度消失"的极端证明。
""")


if __name__ == "__main__":
    check_rnn()
    check_lstm()
    gradient_probe()
    print("\n全部通过: 手写实现与 PyTorch 输出对齐；梯度实验展示 RNN vs LSTM 长程差异。")
