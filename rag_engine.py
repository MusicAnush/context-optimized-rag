import os
import re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI

class StructuralRAGEngine:
    def __init__(self, openai_api_key: str, embedding_model: str = "text-embedding-3-small", llm_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # Storage
        self.chunks = []
        self.tokenized_chunks = []
        self.bm25 = None
        self.index = None
        self.vector_dim = 1536 # Default for text-embedding-3-small

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generates dense vector embeddings via OpenAI API."""
        response = self.client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def ingest_document(self, text: str, chunk_size: int = 500, overlap: int = 100):
        """Chunks text structurally, then builds both FAISS and BM25 indices."""
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Sliding window chunking
        words = text.split(' ')
        self.chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                self.chunks.append(chunk)
                
        if not self.chunks:
            return

        # 1. Initialize Sparse BM25 Index
        self.tokenized_chunks = [doc.lower().split(" ") for doc in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

        # 2. Initialize Dense FAISS Index (HNSW for production scaling style)
        embeddings = []
        for chunk in self.chunks:
            embeddings.append(self._get_embedding(chunk))
        
        embeddings_matrix = np.array(embeddings, dtype=np.float32)
        
        # Using HNSW index for high-performance approximate nearest neighbor search
        self.index = faiss.IndexHNSWFlat(self.vector_dim, 32)
        self.index.add(embeddings_matrix)

    def _reciprocal_rank_fusion(self, dense_results: list, sparse_results: list, k: int = 60, top_n: int = 3) -> list:
        """Calculates Reciprocal Rank Fusion (RRF) scores to combine hybrid streams."""
        rrf_scores = {}
        
        # dense_results/sparse_results are lists of integers (chunk indices)
        for rank, idx in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank + 1))
            
        for rank, idx in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank + 1))
            
        # Sort chunks by highest RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return sorted_indices[:top_n]

    def compress_context(self, retrieved_chunks: list, query: str) -> str:
        """
        Optimizes context by prioritizing structural information density,
        mitigating the 'lost in the middle' phenomenon.
        """
        # A simple, robust compression strategy: strip repetitive phrasing, 
        # keep sentences with high density of query keywords.
        query_terms = set(query.lower().split())
        compressed_blocks = []
        
        for chunk in retrieved_chunks:
            sentences = chunk.split(". ")
            scored_sentences = []
            for sent in sentences:
                score = sum(1 for term in query_terms if term in sent.lower())
                scored_sentences.append((score, sent))
            
            # Sort sentences by keyword relevance to drop completely irrelevant sentences
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            # Reconstruct chunk using the top 75% most descriptive sentences
            keep_count = max(1, int(len(scored_sentences) * 0.75))
            optimized_chunk = ". ".join([s[1] for s in scored_sentences[:keep_count]])
            compressed_blocks.append(optimized_chunk)
            
        return "\n\n---\n\n".join(compressed_blocks)

    def query(self, query_text: str) -> tuple:
        """Executes hybrid search, compresses the context window, and executes generation."""
        if not self.index or not self.bm25:
            return "Engine index is empty. Please ingest a document first.", []

        # Stream 1: Dense Retrieval
        query_vector = self._get_embedding(query_text).reshape(1, -1)
        _, dense_indices = self.index.search(query_vector, k=5)
        dense_res = dense_indices[0].tolist()

        # Stream 2: Sparse Retrieval
        tokenized_query = query_text.lower().split(" ")
        sparse_scores = self.bm25.get_scores(tokenized_query)
        sparse_res = np.argsort(sparse_scores)[::-1][:5].tolist()

        # Fusion
        fused_indices = self._reciprocal_rank_fusion(dense_res, sparse_res, top_n=3)
        raw_retrieved = [self.chunks[idx] for idx in fused_indices]
        
        # Context Optimization Phase
        optimized_context = self.compress_context(raw_retrieved, query_text)

        # Synthesis System Prompt
        system_prompt = (
            "You are a precise AI Research Engineer. Answer the user query using ONLY the verified structural "
            "context provided below. If the answer cannot be determined directly from the context, state that "
            "you do not have sufficient data.\n\n"
            f"CONTEXT:\n{optimized_context}"
        )

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_text}
            ],
            temperature=0.0 # Deterministic responses only
        )

        return response.choices[0].message.content, raw_retrieved
