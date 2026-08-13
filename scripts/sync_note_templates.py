# -*- coding: utf-8 -*-
"""从 docs/_template.md 派生各笔记叶子目录的 <目录>-note.md 可填充模板。

用法:
    python scripts/sync_note_templates.py           # 同步（幂等，只更新有差异的文件）
    python scripts/sync_note_templates.py --check   # 只检查：若有文件需更新则退出码 1（供 CI / 手动校验）

设计:
- 以 docs/_template.md 为唯一结构来源，模板结构变更后运行本脚本即可一键同步全部子目录。
- 主题元数据（中文名 / tags / 关联相邻目录）由 THEMES 提供；新增目录不在映射中时
  用目录名兜底，并在输出中提示补充映射。
- 幂等：生成内容与现有文件一致时跳过，不产生无谓变更。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "docs"
BASE = ROOT / "_template.md"

# 目录名 -> (中文主题名, 标签列表)。新增目录请在此补充映射。
THEMES: dict[str, tuple[str, list[str]]] = {
    "Language-Features": ("语言特性", ["Python", "语言特性"]),
    "Concurrency": ("并发编程", ["Python", "并发编程"]),
    "Packaging": ("打包与依赖", ["Python", "打包"]),
    "Type-Hints": ("类型注解", ["Python", "类型注解"]),
    "Linear-Models": ("线性模型", ["机器学习", "线性模型"]),
    "Tree-Based": ("树模型", ["机器学习", "树模型"]),
    "Clustering": ("聚类", ["机器学习", "聚类"]),
    "Evaluation": ("模型评估", ["机器学习", "评估"]),
    "Fundamentals": ("深度学习基础", ["深度学习"]),
    "PyTorch": ("PyTorch 框架", ["深度学习", "PyTorch"]),
    "TensorFlow": ("TensorFlow 框架", ["深度学习", "TensorFlow"]),
    "Model-Zoo": ("经典模型库", ["深度学习", "模型"]),
    "Text-Preprocessing": ("文本预处理", ["NLP", "预处理"]),
    "Word-Embedding": ("词嵌入", ["NLP", "Embedding"]),
    "Seq2Seq-Attention": ("Seq2Seq 与注意力", ["NLP", "Seq2Seq", "注意力"]),
    "LLM": ("大语言模型", ["LLM"]),
    "Image-Processing": ("图像处理", ["CV", "图像处理"]),
    "CNN-Architectures": ("CNN 架构", ["CV", "CNN"]),
    "Object-Detection": ("目标检测", ["CV", "目标检测"]),
    "Agent-Architecture": ("Agent 架构", ["AI-Agent"]),
    "RAG": ("RAG 检索增强生成", ["RAG"]),
    "Prompt-Engineering": ("提示工程", ["Prompt"]),
    "Frameworks": ("Agent 框架", ["AI-Agent", "框架"]),
    "Use-Cases": ("应用用例", ["AI-Agent", "用例"]),
    "ONNX-Conversion": ("ONNX 转换", ["MLOps", "ONNX"]),
    "Triton-Server": ("Triton 推理服务", ["MLOps", "Triton"]),
    "FastAPI-Serving": ("FastAPI 服务化", ["MLOps", "FastAPI"]),
    "Docker-K8s": ("Docker 与 Kubernetes", ["MLOps", "Docker"]),
    "Sorting": ("排序算法", ["算法", "排序"]),
    "Searching": ("搜索算法", ["算法", "搜索"]),
    "DP-Greedy": ("动态规划与贪心", ["算法", "DP"]),
    "Graph": ("图算法", ["算法", "图"]),
    "Network": ("计算机网络", ["网络"]),
    "OS": ("操作系统", ["OS"]),
    "Database": ("数据库", ["数据库"]),
}

# 3.3 关联知识网占位行（必须与 docs/_template.md 精确一致）
PRE = "- **前置依赖**：[前置笔记](xxx.md)（必先懂什么？）"
LAT = "- **横向关联**：[相关技术笔记](xxx.md)（异同对比？）"
NXT = "- **进阶方向**：[高级笔记](xxx.md)（下一步学什么？）"


def find_note_dirs() -> list[Path]:
    """自动发现所有笔记叶子目录（板块下的一级子目录，排除 assets 类目录）。"""
    dirs: list[Path] = []
    for section in sorted(ROOT.iterdir()):
        if not section.is_dir() or not section.name[0].isdigit():
            continue
        for sub in sorted(section.iterdir()):
            if sub.is_dir() and sub.name not in {"assets"}:
                dirs.append(sub)
    return dirs


def render_template(d: Path) -> tuple[str, bool]:
    """生成目录 d 的模板内容。返回 (内容, 是否使用兜底主题)。"""
    base = BASE.read_text(encoding="utf-8")
    section = d.parent.name
    theme, tags = THEMES.get(d.name, (d.name, [d.name]))
    fallback = d.name not in THEMES

    content = base
    content = content.replace('title: "[笔记标题]"', f'title: "[{theme}：笔记标题]"')
    content = content.replace(
        'description: "[SEO 一句话描述]"',
        f'description: "[{theme}方向——SEO 一句话描述]"',
    )
    content = content.replace("tags: [标签1, 标签2]", f"tags: [{', '.join(tags)}]")

    # 同板块相邻目录（用于 3.3 前置/进阶占位链接）
    siblings = [x for x in sorted(d.parent.iterdir()) if x.is_dir()]
    if len(siblings) > 1:
        i = siblings.index(d)
        before, after = siblings[i - 1], siblings[(i + 1) % len(siblings)]
        before_theme = THEMES.get(before.name, (before.name,))[0]
        after_theme = THEMES.get(after.name, (after.name,))[0]
        pre_line = PRE.replace("[前置笔记](xxx.md)", f"[{before_theme}](../{before.name}/xxx.md)")
        nxt_line = NXT.replace("[高级笔记](xxx.md)", f"[{after_theme}](../{after.name}/xxx.md)")
    else:
        pre_line = PRE.replace("[前置笔记](xxx.md)", f"[{theme}](../{d.name}/xxx.md)")
        nxt_line = NXT.replace("[高级笔记](xxx.md)", f"[{theme}](../{d.name}/xxx.md)")
    content = content.replace(PRE, pre_line).replace(NXT, nxt_line)
    return content, fallback


def main() -> int:
    check_only = "--check" in sys.argv
    if not BASE.exists():
        print(f"错误: 未找到模板 {BASE}", file=sys.stderr)
        return 2

    dirs = find_note_dirs()
    changed: list[str] = []
    fallbacks: list[str] = []
    for d in dirs:
        content, fallback = render_template(d)
        if fallback:
            fallbacks.append(d.name)
        target = d / f"{d.name.lower()}-note.md"
        if target.exists() and target.read_text(encoding="utf-8") == content:
            continue
        if check_only:
            changed.append(str(target.relative_to(ROOT)))
        else:
            target.write_text(content, encoding="utf-8")
            changed.append(str(target.relative_to(ROOT)))

    print(f"发现 {len(dirs)} 个笔记目录")
    if fallbacks:
        print(f"⚠ 以下目录未在 THEMES 中配置，已用目录名兜底（建议补充映射）: {', '.join(fallbacks)}")
    if changed:
        print(f"{'需更新' if check_only else '已更新'} {len(changed)} 个文件:")
        for f in changed:
            print(f"  - {f}")
        return 1 if check_only else 0
    print("✓ 全部模板已是最新，无需更新")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
