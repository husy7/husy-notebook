#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
嵌入模块可用性测试脚本
用途：检查 EMBED_MODEL_TYPE 等环境变量是否生效，模型能否正常加载并编码文本。
"""

import os
import sys
from dotenv import load_dotenv

# 加载 .env（默认当前目录，可指定路径）
load_dotenv()  # 若 .env 不在当前目录，可改为 load_dotenv("绝对路径/.env")

# 打印当前环境变量（脱敏）
print("=" * 50)
print("环境变量检查")
print(f"EMBED_MODEL_TYPE = {os.getenv('EMBED_MODEL_TYPE', '未设置')}")
print(f"EMBED_MODEL_NAME = {os.getenv('EMBED_MODEL_NAME', '未设置')}")
print(f"EMBED_API_KEY    = {'已设置' if os.getenv('EMBED_API_KEY') else '未设置'}")
print(f"EMBED_BASE_URL   = {os.getenv('EMBED_BASE_URL', '未设置')}")
print("=" * 50)

try:
    # 导入模块（假设模块路径在 sys.path 中，或直接复制到当前目录）
    # 如果您的嵌入模块在 hello_agents 包中，可改为：
    # from hello_agents.memory.embedding import get_text_embedder, get_dimension
    # 这里假设文件名为 embedding_provider.py，我们直接导入同目录下的模块
    from embedding import get_text_embedder, get_dimension, refresh_embedder
except ImportError as e:
    print(f"❌ 无法导入嵌入模块：{e}")
    print("请确保 embedding_provider.py 在 Python 路径中，或修改 import 语句。")
    sys.exit(1)

# 尝试获取嵌入器
print("🔄 正在初始化嵌入模型...")
try:
    embedder = get_text_embedder()
    dim = get_dimension()
    print(f"✅ 嵌入模型初始化成功！")
    print(f"   模型类型：{embedder.__class__.__name__}")
    print(f"   向量维度：{dim}")
except Exception as e:
    print(f"❌ 嵌入模型初始化失败：{e}")
    print("\n可能的原因：")
    print("  - 环境变量配置错误（检查 EMBED_MODEL_TYPE 等）")
    print("  - 依赖库未安装（dashscope / sentence-transformers / scikit-learn）")
    print("  - API Key 无效或网络问题")
    sys.exit(1)

# 测试编码
test_texts = ["Hello world", "这是一个测试句子"]
print("\n🔄 测试编码文本...")
try:
    vectors = embedder.encode(test_texts)
    if isinstance(vectors, list) and len(vectors) == len(test_texts):
        print(f"✅ 编码成功！得到 {len(vectors)} 个向量，每个维度为 {len(vectors[0])}")
        # 打印第一个向量的前5个数值
        print(f"   第一个向量前5维：{vectors[0][:5]}")
    else:
        print(f"⚠️ 编码返回结果异常：{type(vectors)}，长度 {len(vectors) if hasattr(vectors, '__len__') else '?'}")
except Exception as e:
    print(f"❌ 编码测试失败：{e}")
    sys.exit(1)

print("\n🎉 所有测试通过，嵌入模型工作正常！")