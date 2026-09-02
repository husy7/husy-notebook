"""GRPO 与 DPO 的玩具实现（PyTorch CPU，可直接运行，几秒出结果）

Part A  GRPO 训练闭环：一个微型"语言策略"（按问题 q 逐位生成 4 个 token），
        用**可验证奖励**（与目标序列的匹配度）打分，组内标准化得优势，
        不带 critic、带 KL(θ‖ref) 惩罚 —— 与 DeepSeek-R1 的 RLVR+GRPO 同构。
Part B  DPO 损失最小演示：证明它是"纯离线监督式"优化。

配套笔记：GRPO与偏好优化-note.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

N_Q = 2          # 两类"问题"
LEN = 4          # 输出序列长度
VOCAB = 6        # 词表大小
TARGETS = {      # 每类问题的"正确答案"（规则可验证）
    0: [1, 2, 3, 4],
    1: [5, 4, 3, 2],
}
DEVICE = "cpu"


class TinyPolicy(nn.Module):
    """q -> 每位置一个独立 Categorical(词表) —— 玩具版"策略 πθ(seq|q)"。"""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(N_Q, hidden),
            nn.ReLU(),
            nn.Linear(hidden, LEN * VOCAB),      # 展平成 (LEN, VOCAB) 的 logits
        )

    def logits(self, q):
        out = self.net(q)                        # (B, LEN*VOCAB)
        return out.view(-1, LEN, VOCAB)          # (B, LEN, VOCAB)

    def sample_and_logp(self, q, g):
        """对 q 的 g 个独立 rollout：返回采样 token (g,LEN) 与逐样本对数概率。"""
        qq = q.repeat_interleave(g)              # 同一问题采样 g 次
        logits = self.logits(qq)                 # (g, LEN, VOCAB)
        dist = torch.distributions.Categorical(logits=logits)
        tokens = dist.sample()                   # (g, LEN)
        logp = dist.log_prob(tokens).sum(-1)     # 整条序列的对数概率
        return tokens, logp


def verifiable_reward(tokens: torch.Tensor, q: int) -> torch.Tensor:
    """规则奖励：每个位置命中目标得 1/len 分（可验证奖励 RLVR 的玩具版）。"""
    target = torch.tensor(TARGETS[q], device=tokens.device)
    return (tokens == target).float().mean(dim=-1)          # (g,)


def grpo_step(policy, ref, opt, q: int, g: int = 16, beta: float = 0.05):
    """一次 GRPO 更新：
       1) 采样 G 个回答 → 规则奖励；
       2) A_i = (r_i - mean)/std(组内标准化, 免 critic)；
       3) 目标 = -A·logπθ + β·KL(θ‖ref)（用样本近似 KL）。"""
    tokens, logp = policy.sample_and_logp(torch.tensor([q]), g)
    r = verifiable_reward(tokens, q)                        # (g,)
    adv = (r - r.mean()) / (r.std() + 1e-4)                 # 组内相对优势
    with torch.no_grad():                                   # ref 冻结
        ref_logp = ref.sample_and_logp(torch.tensor([q]), g)[1]
    loss = -(adv.detach() * logp).mean()                    # 策略梯度项
    loss = loss + beta * (logp - ref_logp).mean()           # KL 惩罚(样本近似)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return r.mean().item(), loss.item()


def run_grpo(iterations: int = 600, g: int = 16):
    print("== Part A: GRPO(组内相对优势 + 规则奖励 + KL) 训练微型策略 ==")
    policy = TinyPolicy().to(DEVICE)
    ref = TinyPolicy().to(DEVICE)
    ref.load_state_dict(policy.state_dict())                # π_ref = 初始策略
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=5e-3)

    q = 0
    for step in range(1, iterations + 1):
        q = step % N_Q                       # 两类问题交替训练
        r, loss = grpo_step(policy, ref, opt, q, g)
        if step % 150 == 0:
            print(f"  step {step:4d}  loss={loss:.3f}  本步平均奖励={r:.3f}")
    # 贪心解码看学到什么
    with torch.no_grad():
        for qi in range(N_Q):
            logits = policy.logits(torch.tensor([qi]))
            best = logits.argmax(-1)[0].tolist()
            ok = best == TARGETS[qi]
            print(f"  q={qi} 贪心输出={best}  目标={TARGETS[qi]}  {'✓正确' if ok else '✗'}")
    return policy


def demo_dpo():
    print("\n== Part B: DPO 损失（离线、无 RM/在线采样，就是二元分类损失）==")
    # 制造"偏好对"：chosen 是比 rejected 更好的回答（此处用假的逐 token logp 示意）
    beta = 0.1
    logp_chosen, logp_ref_chosen = -0.3, -0.8      # chosen 相对 ref 明显提升
    logp_rej, logp_ref_rej = -2.5, -1.2            # rejected 相对 ref 没有提升
    # L = -log σ( β·( (logp_w - logp_ref_w) - (logp_l - logp_ref_l) ) )
    gap = (logp_chosen - logp_ref_chosen) - (logp_rej - logp_ref_rej)
    loss = -torch.log(torch.sigmoid(torch.tensor(beta * gap)))
    print(f"  chosen 提升量={(logp_chosen - logp_ref_chosen):.2f}  "
          f"rejected 提升量={(logp_rej - logp_ref_rej):.2f}")
    print(f"  DPO 间隔 gap={beta * gap:.2f} > 0 → 损失={loss.item():.3f}（小=好）")
    print("  含义: 让 πθ 相对 π_ref 的提升向 chosen 倾斜——纯监督、无需奖励模型。")


if __name__ == "__main__":
    run_grpo()
    demo_dpo()
