# -*- coding: utf-8 -*-
"""交付检查：对知识库仓库做 F1-F4 全面验证，输出 PASS/FAIL 报告。"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"G:\Aknowledge-base")


def run(cmd: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8")


results: list[tuple[str, bool, str]] = []


def check(desc: str, ok: bool, detail: str = "") -> None:
    results.append((desc, ok, detail))


# ---------- A. Git 与仓库卫生 ----------
r = run(["git", "status", "--porcelain"])
check("A1 git 工作区干净（无未提交/未跟踪文件）", r.stdout.strip() == "", r.stdout.strip())

r = run(["git", "branch", "--show-current"])
check("A2 当前分支为 main", r.stdout.strip() == "main", r.stdout.strip())

r = run(["git", "log", "--oneline"])
prefixes = re.findall(r"^\w+ (\w+):", r.stdout, re.M)
bad = [p for p in prefixes if p not in {"docs", "ci", "code"}]
check("A3 提交信息前缀符合约定（docs:/ci:/code:）", not bad, f"违规前缀: {bad or '无'}")

r = run(["git", "ls-files"])
forbidden = [f for f in r.stdout.splitlines() if f.startswith(("site/", ".codegraph/", "__pycache__/", ".omo/", ".reasonix/"))]
check("A4 site/.codegraph/.omo/.reasonix 未入库", not forbidden, f"违规: {forbidden[:3] or '无'}")

gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
required = [".codegraph/", ".omo/", ".reasonix/", "__pycache__/", "*.pyc", ".venv/", "site/", ".DS_Store"]
missing = [x for x in required if x not in gitignore]
check("A5 .gitignore 覆盖完整", not missing, f"缺失: {missing or '无'}")

# ---------- B. 结构完整性 ----------
sections = [d.name for d in (ROOT / "docs").iterdir() if d.is_dir() and d.name[0].isdigit()]
boards = [s for s in sections if s != "00-Index"]
check("B1 知识地图 00-Index + 10 大板块齐全",
      "00-Index" in sections and len(boards) == 10, f"00-Index + {len(boards)} 板块: {boards}")

missing_index = [s for s in sections if not (ROOT / "docs" / s / "index.md").exists()]
check("B2 每板块 index.md 存在", not missing_index, f"缺失: {missing_index or '无'}")

note_dirs = [d for s in sections for d in (ROOT / "docs" / s).iterdir() if d.is_dir()]
missing_gk = [d.name for d in note_dirs if not (d / ".gitkeep").exists()]
check("B3 叶子目录 .gitkeep 齐全", not missing_gk, f"缺失: {missing_gk or '无'}")
check("B4 叶子目录数量为 35", len(note_dirs) == 35, str(len(note_dirs)))

tpl = (ROOT / "docs" / "_template.md").read_text(encoding="utf-8")
check("B5 _template.md 无 frontmatter 前导空行", tpl.startswith("---"), "首行: " + repr(tpl.splitlines()[0]))

# ---------- C. 模板同步 ----------
r = run([sys.executable, "scripts/sync_note_templates.py", "--check"])
check("C1 35 个 -note.md 与模板同步", r.returncode == 0, r.stdout.strip().splitlines()[-1])

# ---------- D. 构建与站点 ----------
r = run([sys.executable, "-m", "mkdocs", "build", "--clean"])
warn = [l for l in r.stdout.splitlines() + r.stderr.splitlines() if "WARNING" in l or "ERROR" in l]
check("D1 mkdocs build 退出码 0", r.returncode == 0, f"exit={r.returncode}")
check("D2 构建零警告", not warn, "\n".join(warn[:3]) or "无")

site = ROOT / "site"
leak = [p for p in site.rglob("index.html") if "_template" in str(p) or "-note" in str(p)]
check("D3 _template/-note 模板未进入站点", not leak, f"泄漏: {[str(p) for p in leak[:3]] or '无'}")
page_count = len(list(site.glob("*/index.html")))
check("D4 站点含 10 个板块页面", page_count >= 10, f"板块页: {page_count}")

html = (site / "index.html").read_text(encoding="utf-8")
check("D5 MathJax 注入（mathjax.js）", "mathjax" in html, "index.html 含 mathjax 引用")

# ---------- E. 代码质量（F2） ----------
r = run([sys.executable, "-m", "py_compile", "scripts/sync_note_templates.py"])
check("E1 sync 脚本 py_compile 通过", r.returncode == 0, r.stderr.strip() or "ok")

script = (ROOT / "scripts" / "sync_note_templates.py").read_text(encoding="utf-8")
no_anno = re.findall(r"^def (\w+)\([^)]*\)(?=:)", script, re.M)
check("E2 脚本函数均带返回类型注解", not no_anno, f"无返回注解: {no_anno or '无'}")
check("E3 脚本含 --check 模式", "--check" in script, "")

# ---------- F. 规范一致性 ----------
r = run(["grep", "-rn", r"\[\[", "docs", "--include=*.md"], cwd=ROOT)
wikilinks = [l for l in r.stdout.splitlines() if "Callable" not in l and "双链" not in l]
check("F1 无 [[双链]] 残留（排除代码误报）", not wikilinks, "\n".join(wikilinks[:3]) or "无")

r = run(["grep", "-rn", "费曼五步法", "docs", "AGENTS.md", "README.md", "--include=*.md"], cwd=ROOT)
check("F2 无费曼五步法旧描述残留", r.returncode != 0 or r.stdout.strip() == "", r.stdout.strip() or "无")

# ---------- G. 范围保真（F4） ----------
r = run(["git", "ls-files"])
junk = [f for f in r.stdout.splitlines() if ".tmp" in f or ".verify" in f]
check("G1 无临时/验证文件入库", not junk, f"违规: {junk or '无'}")
root_files = sorted(p.name for p in ROOT.iterdir() if p.is_file())
unexpected = [f for f in root_files if not f.startswith((".git",)) and f not in
              {".gitignore", "AGENTS.md", "README.md", "mkdocs.yml"}]
check("G2 仓库根目录无意外文件", not unexpected, f"意外: {unexpected or '无'}")

# ---------- 汇总 ----------
print("=" * 60)
for desc, ok, detail in results:
    print(f"{'✅' if ok else '❌'} {desc}" + (f" — {detail}" if detail else ""))
print("=" * 60)
failed = [d for d, ok, _ in results if not ok]
print(f"总计 {len(results)} 项: {'✅ 全部通过' if not failed else f'❌ {len(failed)} 项失败: {failed}'}")
sys.exit(1 if failed else 0)
