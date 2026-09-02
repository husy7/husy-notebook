"""提示学习与上下文学习（ICL）脚手架演示（默认零依赖，可直接运行）

真正的大模型 ICL 行为 = "在示例约束下的概率续写"，需要在 transformers/vLLM +
真实模型权重上跑。本脚本为了"任何环境都能跑通"：

  1) 用 LLM 服务的接口封装一个 complete(prompt)：有本地 HF 模型时可切换真实生成
     （见 _hf_complete，环境变量 USE_HF=1 + 已装 transformers），默认走 _mock_complete：
     一个"按与示例词重叠度挑最近演示"的启发式模拟器，用于演示**提示模板/示例
     选择/0-shot vs few-shot** 这些机制层面的差异（结果数值无真实语义）；
  2) 对比 zero-shot 与 few-shot 在同一组测试集上的表现，打印构造出的提示原文；
  3) 演示示例顺序/选取对输出的敏感度。

配套笔记：提示学习与上下文学习-note.md
"""
import os
import re

# ---------------- 玩具数据：电影评论情感二分类 ----------------
DEMO_POOL = [          # 少量"标注示例"库，few-shot 从这里选
    ("这部电影节奏明快笑点密集", "POSITIVE"),
    ("导演功力深厚画面太美了", "POSITIVE"),
    ("主角演技在线故事非常感人", "POSITIVE"),
    ("配乐和剪辑都堪称惊艳", "POSITIVE"),
    ("剧情拖沓看得昏昏欲睡", "NEGATIVE"),
    ("剧本漏洞百出毫无逻辑", "NEGATIVE"),
    ("特效粗糙像五毛钱水平", "NEGATIVE"),
    ("台词尴尬表演浮夸至极", "NEGATIVE"),
]
TEST_SET = [
    ("剧情紧凑全程无尿点", "POSITIVE"),       # 金标随样本携带，便于评估
    ("布景廉价表演十分生硬", "NEGATIVE"),
    ("笑点密集非常适合全家观看", "POSITIVE"),
    ("剪辑混乱让我中途离场", "NEGATIVE"),
    ("配乐空灵画面极具诗意", "POSITIVE"),
]

SYSTEM = "你是情感分析助手。只输出 POSITIVE 或 NEGATIVE，不要输出其他内容。"


# ---------------- "LLM" 封装 ----------------
def _mock_complete(prompt: str) -> str:
    """零依赖模拟器：取查询句与示例句的最大字符重叠，返回对应示例的标签；
    无任何示例命中时回退到"示例多数标签"。

    这是 nearest-neighbor 启发式，只用来演示模板机制：它"照搬最像示例的输出"，
    恰恰放大了 ICL 的已知缺陷——示例标签失衡/缺失时结果被示例带偏。
    真实 ICL 由模型的预训练条件概率完成（见 _hf_complete）。"""
    query, demo_labels, best, best_score = "", [], None, -1
    for ln in reversed(prompt.splitlines()):          # 查询行带 Review: 且在最后
        if not query:
            m = re.search(r"Review:\s*(.+)", ln)
            if m:
                query = m.group(1)
        m = re.search(r"Review:\s*(.+?)\s*→\s*(\w+)", ln)
        if m:
            demo_labels.append(m.group(2))            # 记录示例标签 → 多数回退
            score = len(set(query) & set(m.group(1)))
            if score > best_score:
                best, best_score = m.group(2), score
    if best is not None:
        return best
    if demo_labels:                                   # 多数标签回退
        return max(set(demo_labels), key=demo_labels.count)
    return "?"                                        # 连示例都没有：无法对齐格式


def _hf_complete(prompt: str, max_new: int = 8) -> str:
    """真实 LLM 路径（可选）：本地 tiny 模型演示，需先 pip install transformers。
    用 pip 装好并设 USE_HF=1 后自动启用：USE_HF=1 python 提示学习与上下文学习_sample.py"""
    from transformers import pipeline      # 延迟 import，缺库时不影响默认演示
    gen = pipeline("text-generation", model="sshleifer/tiny-gpt2")
    return gen(prompt, max_new_tokens=max_new, do_sample=False)[0]["generated_text"][len(prompt):]


def complete(prompt: str) -> str:
    if os.environ.get("USE_HF") == "1":
        try:
            return _hf_complete(prompt)
        except Exception as e:            # 模型下载失败/缺库 → 退回模拟器并说明
            print(f"[提示] 真实模型不可用({e})，退回模拟器")
    return _mock_complete(prompt)


# ---------------- 提示模板 ----------------
def make_zero_shot_prompt(query: str) -> str:
    return f"{SYSTEM}\n\nReview: {query}\nSentiment:"


def make_few_shot_prompt(demos, query: str) -> str:
    lines = [SYSTEM, "", "下面是几个示例："]
    for text, label in demos:
        lines.append(f"Review: {text} → {label}")
    lines += ["", f"Review: {query}", "Sentiment:"]
    return "\n".join(lines)


def parse_answer(raw: str) -> str:
    """从模型输出里抠出标签；容忍模型多输出几个词（解析容错演示）。"""
    raw = raw.strip()
    for label in ("POSITIVE", "NEGATIVE"):
        if label in raw:
            return label
    # 找不到明确标签 → 取 Sentiment: 之后第一个词（模拟"格式漂移"场景）
    tail = raw.split("Sentiment:")[-1].strip().split()
    return tail[0][:16] if tail else "?"


# ---------------- 评估与展示 ----------------
def eval_accuracy(build_prompt) -> tuple:
    hits = 0
    details = []
    for text, label in TEST_SET:
        pred = parse_answer(complete(build_prompt(text)))
        ok = pred == label
        hits += ok
        details.append((text, label, pred, ok))
    return hits / len(TEST_SET), details


def pick_demos(query: str, k: int = 4):
    """示例选择：按与查询的重叠度挑最相关的 k 条（模拟"精选演示"）。"""
    q = set(query)
    return sorted(DEMO_POOL, key=lambda d: -len(q & set(d[0])))[:k]


def main() -> None:
    # 1) 打印一条构造出的 few-shot 提示（看模板长什么样）
    print("== 构造出的 few-shot 提示示例 ==")
    prompt = make_few_shot_prompt(DEMO_POOL[:4], "剧情紧凑全程无尿点")
    print(prompt)
    print(f"\n→ 模型输出: {complete(prompt)!r}")

    # 2) 0-shot vs few-shot（固定前 4 条 = 全正例 → 演示"示例标签失衡"）对比
    acc0, det0 = eval_accuracy(make_zero_shot_prompt)
    acc1, det1 = eval_accuracy(lambda q: make_few_shot_prompt(DEMO_POOL[:4], q))
    acc2, det2 = eval_accuracy(lambda q: make_few_shot_prompt(pick_demos(q, 4), q))
    print(f"\n== 效果对比 (零依赖模拟器；数值只用于说明机制趋势) ==")
    print(f"  zero-shot  : {acc0:.0%}   (无示例可抄 → 模拟器无法对齐格式，多半输出'?')")
    print(f"  4-shot固定 : {acc1:.0%}   (DEMO_POOL 前 4 条全是正例 → 示例标签失衡)")
    print(f"  4-shot精选 : {acc2:.0%}   (按与查询相关度选 4 条，正负覆盖均衡)")
    print("""
  注意: 真实 LLM 的 zero-shot 远强于此（预训练知识足以理解"情感分析"指令）；
        模拟器只能"抄最近示例"，所以它放大了 ICL 的脆弱面。真实 ICL 已知的
        敏感因素: 示例顺序颠倒、示例标签分布、措辞/分隔符格式——评测时请多次
        随机化示例顺序取均值，别被单次结果骗了。""")

    # 4) 逐条看"精选 few-shot"预测明细（观察失败模式）
    print("\n== 4-shot 精选 逐条明细 ==")
    for text, gold, pred, ok in det2:
        print(f"  {'✓' if ok else '✗'}  金标={gold:<8} 预测={pred:<10} {text}")


if __name__ == "__main__":
    main()
