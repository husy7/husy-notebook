"""rag/pipeline.py 纯逻辑测试（不触发网络/模型调用）。

只覆盖 chunking、去重、文本预处理、片段合并等纯函数。
"""
import unittest

from rag.pipeline import (
    _split_paragraphs_with_headings,
    _chunk_paragraphs,
    _dedupe_chunks,
    _is_cjk,
    _approx_token_len,
    _preprocess_markdown_for_embedding,
    merge_snippets,
)


class CjkTest(unittest.TestCase):

    def test_is_cjk(self):
        self.assertTrue(_is_cjk("中"))
        self.assertFalse(_is_cjk("a"))


class ApproxTokenLenTest(unittest.TestCase):

    def test_chinese_count(self):
        # "你好世界": 4 CJK + 整个字符串作为1个无空格 token = 5
        self.assertEqual(_approx_token_len("你好世界"), 5)

    def test_mixed(self):
        # "hello 世界": 2 CJK + "hello"、"世界" 两个 token = 4
        self.assertEqual(_approx_token_len("hello 世界"), 4)


class SplitParagraphsTest(unittest.TestCase):

    def test_plain_text(self):
        paras = _split_paragraphs_with_headings("第一段\n\n第二段")
        # 空行会 flush 段落
        self.assertGreaterEqual(len(paras), 2)

    def test_heading_path(self):
        text = "# 标题甲\n内容甲\n\n## 标题乙\n内容乙"
        paras = _split_paragraphs_with_headings(text)
        # 标题甲 段落带 heading_path
        heading_paras = [p for p in paras if p.get("heading_path")]
        self.assertGreaterEqual(len(heading_paras), 1)

    def test_content_empty_returns_single(self):
        paras = _split_paragraphs_with_headings("   ")
        self.assertEqual(len(paras), 1)


class ChunkParagraphsTest(unittest.TestCase):

    def test_chunk_respects_size(self):
        # 每个段落用大量带空格的词，使 token 估算显著增长，从而触发分块
        paras = [{"content": " ".join(["word"] * 200), "start": i * 200, "end": i * 200 + 800} for i in range(5)]
        chunks = _chunk_paragraphs(paras, chunk_tokens=300, overlap_tokens=50)
        self.assertGreaterEqual(len(chunks), 2)

    def test_small_chunks_unchanged(self):
        paras = [{"content": "ab", "start": 0, "end": 2}]
        chunks = _chunk_paragraphs(paras, chunk_tokens=100, overlap_tokens=0)
        self.assertEqual(len(chunks), 1)


class DedupeChunksTest(unittest.TestCase):

    def test_global_dedup(self):
        c1 = {"content_hash": "h1", "content": "a"}
        c2 = {"content_hash": "h1", "content": "a"}
        c3 = {"content_hash": "h2", "content": "b"}
        out = _dedupe_chunks([[c1], [c2, c3]])
        self.assertEqual(len(out), 2)


class PreprocessMarkdownTest(unittest.TestCase):

    def test_strips_headers(self):
        t = _preprocess_markdown_for_embedding("# 标题\n正文")
        self.assertNotIn("#", t.split("\n")[0])
        self.assertIn("标题", t)

    def test_strips_link_markers_keeps_text(self):
        t = _preprocess_markdown_for_embedding("[文本](url)")
        self.assertIn("文本", t)
        self.assertNotIn("](url)", t)


class MergeSnippetsTest(unittest.TestCase):

    def test_merge_caps_at_max(self):
        items = [
            {"content": "a" * 100},
            {"content": "b" * 100},
        ]
        out = merge_snippets(items, max_chars=150)
        self.assertLessEqual(len(out), 150 + 3)  # 两个换行分隔


if __name__ == "__main__":
    unittest.main()
