import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import faiss
from rag_engine import StructuralRAGEngine

class TestStructuralRAGEngine(unittest.TestCase):

    @patch('rag_engine.OpenAI')
    def setUp(self, mock_openai):
        """Sets up the RAG engine with a mocked OpenAI client for testing."""
        self.mock_client = MagicMock()
        mock_openai.return_value = self.mock_client
        
        # Initialize engine with a placeholder key
        self.engine = StructuralRAGEngine(openai_api_key="fake-key")
        
        # Define a mock embedding vector matching OpenAI's 1536 dimensions
        self.dummy_embedding = [0.1] * 1536

    @patch('rag_engine.OpenAI')
    def test_get_embedding(self, mock_openai):
        """Validates that _get_embedding calls the OpenAI API and returns a formatted NumPy array."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=self.dummy_embedding)]
        self.engine.client.embeddings.create.return_value = mock_response

        embedding = self.engine._get_embedding("Test string")
        
        # Verify call parameters
        self.engine.client.embeddings.create.assert_called_once_with(
            input=["Test string"],
            model="text-embedding-3-small"
        )
        # Check types, precision, and shape constraints
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.dtype, np.float32)
        self.assertEqual(embedding.shape, (1536,))

    @patch('rag_engine.OpenAI')
    def test_ingest_document(self, mock_openai):
        """Verifies that text ingestion chunks data and populates FAISS and BM25 matrices."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=self.dummy_embedding)]
        self.engine.client.embeddings.create.return_value = mock_response

        sample_corpus = (
            "The quick brown fox jumps over the lazy dog. "
            "Artificial intelligence is transforming software engineering paradigms. "
            "Vector databases enable low latency nearest neighbor calculations."
        )
        
        # Ingest with small chunk configuration to enforce separation logic
        self.engine.ingest_document(sample_corpus, chunk_size=10, overlap=2)

        # Assert arrays populated successfully
        self.assertGreater(len(self.engine.chunks), 0)
        self.assertIsNotNone(self.engine.bm25)
        self.assertIsInstance(self.engine.index, faiss.IndexHNSWFlat)
        
        # Verify index count lines up with parsed slice components
        self.assertEqual(self.engine.index.ntotal, len(self.engine.chunks))

    def test_reciprocal_rank_fusion(self):
        """Validates that RRF logic correctly evaluates intersecting index positions."""
        dense_res = [0, 1, 2]
        sparse_res = [1, 0, 3]

        fused_results = self.engine._reciprocal_rank_fusion(dense_res, sparse_res, k=60, top_n=2)

        # Elements overlapping on high rank coordinates should win placement bounds
        self.assertEqual(len(fused_results), 2)
        self.assertIn(0, fused_results)
        self.assertIn(1, fused_results)

    def test_compress_context(self):
        """Ensures text window optimization filters fragments with zero keyword intersections."""
        retrieved_chunks = [
            "Deep learning relies on backpropagation. Weather patterns are highly unpredictable.",
            "Neural networks utilize matrix multiplications."
        ]
        query = "deep learning neural networks"

        compressed = self.engine.compress_context(retrieved_chunks, query)

        # Expected intersections remain untouched
        self.assertIn("Deep learning relies on backpropagation", compressed)
        self.assertIn("Neural networks utilize matrix multiplications", compressed)
        
        # Off-topic phrase lacks query intersections and should be dropped completely
        self.assertNotIn("Weather patterns are highly unpredictable", compressed)

    @patch('rag_engine.OpenAI')
    def test_query_empty_index_guard(self, mock_openai):
        """Ensures system safely handles querying attempts prior to dataset intake steps."""
        answer, chunks = self.engine.query("What is AI?")
        self.assertEqual(chunks, [])
        self.assertIn("Engine index is empty", answer)

if __name__ == '__main__':
    unittest.main()
