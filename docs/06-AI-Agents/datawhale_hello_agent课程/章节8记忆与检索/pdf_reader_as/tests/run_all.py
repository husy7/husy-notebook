"""统一功能测试运行器（标准库，零第三方依赖）。

用法（在项目根目录）：
    python tests/run_all.py            # 运行全部离线单测（默认跳过集成）
    python tests/run_all.py -v         # 详细输出

    RUN_INTEGRATION=1 python tests/run_all.py   # 额外运行真实 Qdrant/LLM 集成测试

说明：
- 项目所有顶层模块（base/rag_tool/memory_tool/...）以绝对导入，需项目根目录在 sys.path。
  本脚本自动将项目根目录加入 sys.path。
"""
import os
import sys
import unittest

# 将项目根目录（tests/ 的上一级）加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 确保当前目录是项目根，便于相对路径模块（如 .env / memory_data）正确加载
if os.path.exists(os.path.join(PROJECT_ROOT, ".env")):
    os.chdir(PROJECT_ROOT)


def main():
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir=start_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2 if "-v" in sys.argv else 1)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
