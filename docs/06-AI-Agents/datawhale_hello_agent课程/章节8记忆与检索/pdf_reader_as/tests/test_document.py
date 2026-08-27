"""rag/document.py 测试：Document, DocumentChunk, DocumentProcessor。"""
import unittest
import tempfile
import os

from rag.document import (
    Document,
    DocumentChunk,
    DocumentProcessor,
    create_document,
    load_text_file,
)


class DocumentTest(unittest.TestCase):

    def test_doc_id_auto_generated(self):
        d = Document(content="hello", metadata={})
        self.assertEqual(len(d.doc_id), 32)  # md5 hex

    def test_doc_id_stable(self):
        d1 = Document(content="same", metadata={})
        d2 = Document(content="same", metadata={})
        self.assertEqual(d1.doc_id, d2.doc_id)

    def test_doc_explicit_id(self):
        d = Document(content="x", metadata={}, doc_id="custom")
        self.assertEqual(d.doc_id, "custom")


class DocumentChunkTest(unittest.TestCase):

    def test_chunk_id_auto(self):
        c = DocumentChunk(content="data", metadata={}, doc_id="doc1", chunk_index=0)
        self.assertEqual(len(c.chunk_id), 32)

    def test_create_document(self):
        d = create_document("content", source="x")
        self.assertEqual(d.content, "content")
        self.assertEqual(d.metadata["source"], "x")


class DocumentProcessorTest(unittest.TestCase):

    def test_short_text_single_chunk(self):
        dp = DocumentProcessor(chunk_size=1000)
        doc = create_document("a" * 100)
        chunks = dp.process_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "a" * 100)

    def test_long_text_multiple_chunks(self):
        dp = DocumentProcessor(chunk_size=200, chunk_overlap=20)
        doc = create_document(("word " * 500))
        chunks = dp.process_document(doc)
        self.assertGreater(len(chunks), 1)
        # 每块必须有 doc_id 与 chunk_index
        for i, c in enumerate(chunks):
            self.assertEqual(c.doc_id, doc.doc_id)
            self.assertEqual(c.chunk_index, i)

    def test_process_documents_batch(self):
        dp = DocumentProcessor(chunk_size=500)
        docs = [create_document("x" * 600), create_document("y" * 50)]
        chunks = dp.process_documents(docs)
        self.assertGreaterEqual(len(chunks), 2)

    def test_merge_chunks(self):
        dp = DocumentProcessor(chunk_size=1000)
        doc = create_document("hello world")
        c1 = DocumentChunk(content="abc", metadata={}, doc_id=doc.doc_id, chunk_index=0)
        c2 = DocumentChunk(content="def", metadata={}, doc_id=doc.doc_id, chunk_index=1)
        merged = dp.merge_chunks([c1, c2], max_length=100)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].content, "abc\ndef")

    def test_filter_chunks(self):
        dp = DocumentProcessor()
        c1 = DocumentChunk(content="short", metadata={})
        c2 = DocumentChunk(content="this is a long enough chunk content", metadata={})
        kept = dp.filter_chunks([c1, c2], min_length=20)
        self.assertEqual([c.content for c in kept], [c2.content])

    def test_add_chunk_metadata(self):
        dp = DocumentProcessor()
        c = DocumentChunk(content="data", metadata={})
        dp.add_chunk_metadata([c], {"tag": "t"})
        self.assertEqual(c.metadata["tag"], "t")


class LoadTextFileTest(unittest.TestCase):

    def test_load_text_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("file content")
            path = f.name
        try:
            d = load_text_file(path)
            self.assertEqual(d.content, "file content")
            self.assertEqual(d.metadata["type"], "text_file")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
