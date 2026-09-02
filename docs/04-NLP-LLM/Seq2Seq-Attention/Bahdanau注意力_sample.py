"""带 Bahdanau（加性）注意力的 Seq2Seq：数字序列"反转"任务（PyTorch，可直接运行）

为什么选反转任务：无注意力的 Seq2Seq 只会把整句压进最后隐状态，根本学不会
"按位置逆序抄写"；加了注意力后解码器每步通过  α_t = softmax(vᵀ tanh(W1·h + W2·s))
自主学会"看源句的哪个位置"——训练后打印对齐矩阵，能看到漂亮的反对角线。

配套笔记：Bahdanau注意力-note.md
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

# ---- 词表：0~9 数字 + SOS/EOS/PAD（PAD 本演示不用，仅作占位约定）----
SOS, EOS, PAD = 10, 11, 12
VOCAB = 13


def make_batch(batch_size: int, max_len: int = 7):
    """一批同长随机数字序列。src: (B,L)；tgt = src 逆序（任务：反转）。"""
    L = random.randint(3, max_len)
    src = torch.randint(0, 10, (batch_size, L))
    tgt = torch.flip(src, dims=[1])                     # 反转 = 目标
    return src, tgt, L


class BahdanauAttention(nn.Module):
    """e_ti = vᵀ tanh(W1·h_i + W2·s_{t-1})；α = softmax(e_t)；c_t = Σ α_i h_i"""

    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 32):
        super().__init__()
        self.W1 = nn.Linear(enc_dim, attn_dim, bias=False)   # 源隐状态 -> 公共空间
        self.W2 = nn.Linear(dec_dim, attn_dim, bias=False)   # decoder 状态 -> 公共空间
        self.v = nn.Linear(attn_dim, 1, bias=False)          # 公共空间 -> 标量分数

    def forward(self, enc_out, s_prev):
        """enc_out: (B, L, enc_dim)（源序列全部隐状态）
           s_prev:  (B, dec_dim)（decoder 上一步状态，Bahdanau 用它打分）
           返回: context (B, enc_dim), alpha (B, L)（对齐权重，可解释性输出）"""
        energy = torch.tanh(self.W1(enc_out) + self.W2(s_prev).unsqueeze(1))  # (B,L,A)
        score = self.v(energy).squeeze(-1)                 # (B, L) 对齐分数
        alpha = torch.softmax(score, dim=-1)               # 对源位置归一化
        ctx = (alpha.unsqueeze(-1) * enc_out).sum(dim=1)   # 软检索：加权和
        return ctx, alpha


class AttnSeq2Seq(nn.Module):
    def __init__(self, vocab=VOCAB, emb=16, enc_hid=24, dec_hid=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        enc_dim = 2 * enc_hid                       # 双向 encoder 拼接
        # 双向 GRU：每个位置能同时看左右，h_i 质量更高
        self.encoder = nn.GRU(emb, enc_hid, bidirectional=True, batch_first=True)
        self.attn = BahdanauAttention(enc_dim, dec_hid)
        # decoder 每步输入 = [词向量; 注意力上下文]，长度 = emb + enc_dim
        self.decoder = nn.GRUCell(emb + enc_dim, dec_hid)
        self.proj = nn.Linear(dec_hid, vocab)       # s_t -> 词表 logits
        self.s0 = nn.Linear(enc_dim, dec_hid)       # encoder 终态 -> s_0

    def encode(self, src):
        emb = self.emb(src)                                  # (B,L,emb)
        out, hn = self.encoder(emb)                          # out (B,L,2H), hn (2,B,H)
        h_fwd, h_bwd = hn[0], hn[1]                          # 首/末方向终态
        s0 = torch.tanh(self.s0(torch.cat([h_fwd, h_bwd], dim=-1)))
        return out, s0

    def forward(self, src, tgt_in, tgt_out):
        """teacher forcing 训练：tgt_in = [SOS] + 真值，tgt_out = 真值 + [EOS]"""
        enc_out, s = self.encode(src)
        B, L = src.shape
        ctx = torch.zeros(B, enc_out.shape[-1], device=src.device)
        logits, losses = [], []
        x = tgt_in[:, 0]                                       # 首步输入 = SOS
        for t in range(tgt_out.shape[1]):
            ctx, alpha = self.attn(enc_out, s)                 # 用 s_{t-1} 对齐
            x_emb = self.emb(x)
            s = self.decoder(torch.cat([x_emb, ctx], dim=-1), s)  # 上下文拼进解码
            logit = self.proj(s)
            logits.append(logit)
            losses.append(F.cross_entropy(logit, tgt_out[:, t]))
            x = tgt_in[:, t + 1] if t + 1 < tgt_in.shape[1] else tgt_in[:, -1]
        return sum(losses) / len(losses), torch.stack(logits)

    @torch.no_grad()
    def translate(self, src, max_len=8):
        """推理：自回归贪心解码，返回 (预测id列表, alpha 矩阵)"""
        enc_out, s = self.encode(src)                            # src 已是 (1,L) 一个 batch
        x = torch.tensor([SOS])
        preds, alphas = [], []
        for _ in range(max_len):
            ctx, alpha = self.attn(enc_out, s)
            s = self.decoder(torch.cat([self.emb(x), ctx], dim=-1), s)
            x = self.proj(s).argmax(-1)
            preds.append(x.item())
            alphas.append(alpha.squeeze(0))
            if x.item() == EOS:
                break
        return preds, alphas


def train_model(steps: int = 1500, batch: int = 32):
    model = AttnSeq2Seq()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(1, steps + 1):
        src, tgt, L = make_batch(batch)
        # tgt_in = [SOS] 前缀真值；tgt_out = 真值 + [EOS]
        tgt_in = torch.cat([torch.full((batch, 1), SOS), tgt], dim=1)
        tgt_out = torch.cat([tgt, torch.full((batch, 1), EOS)], dim=1)
        loss, _ = model(src, tgt_in, tgt_out)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)      # RNN 训练标配
        opt.step()
        if step % 300 == 0:
            print(f"  step {step:5d}  loss = {loss.item():.3f}")
    return model


@torch.no_grad()
def evaluate(model, n: int = 50):
    model.eval()
    ok = 0
    for _ in range(n):
        src, tgt, L = make_batch(1)
        preds, _ = model.translate(src)
        preds = [p for p in preds if p < 10]                   # 去掉 SOS/EOS
        if preds[:L] == tgt[0].tolist():
            ok += 1
    print(f"  推理准确率: {ok}/{n}")
    return ok / n


def show_alignment(model):
    """随机取一条样本：打印 目标位置×源位置 的对齐权重（反转任务应呈反对角线）。"""
    src, tgt, L = make_batch(1)
    preds, alphas = model.translate(src)
    src_toks = [str(t) for t in src[0].tolist()]
    pred_toks = [str(p) for p in preds if p < 10]
    print("\n源序列 :", " ".join(src_toks))
    print("目标   :", " ".join(str(t) for t in tgt[0].tolist()))
    print("预测   :", " ".join(pred_toks))
    # 可视化权重：密度字符表，越亮 = 权重越大
    bar = " .:*#"
    print("\n对齐矩阵 (行=解码步/目标位, 列=源位置, 越亮权重越大):")
    print("        " + "  ".join(f"{s:>2}" for s in src_toks))
    for t, a in enumerate(alphas[:L]):
        row = "".join(bar[min(4, int(v * 5))] for v in a.tolist())
        mark = f"{pred_toks[t]:>2}" if t < len(pred_toks) else "  "
        print(f"  t={t} {mark} | {row}")


if __name__ == "__main__":
    print("训练带 Bahdanau 注意力的 Seq2Seq（任务：反转数字序列）...")
    m = train_model()
    evaluate(m)
    show_alignment(m)
    show_alignment(m)
