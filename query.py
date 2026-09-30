# Actual Application of the RAG pipeline: it is executed by the sever.py
from google import genai
from dotenv import load_dotenv
import os
import time
from generation import build_prompt
from embedding_manager import EmbeddingManager
from retrieval import (
    get_all_chunks_from_db,
    build_bm25_index,
    query_collection,
    bm25_search,
    merge_rrf,
    rerank
)

load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

_SHARED_CROSS_ENCODER = None


def _get_shared_cross_encoder():
    global _SHARED_CROSS_ENCODER
    if _SHARED_CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
            _SHARED_CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            # Fall back gracefully if DLL or package blocked/missing
            _SHARED_CROSS_ENCODER = None
    return _SHARED_CROSS_ENCODER


class RAGPipeline:
    def __init__(self):
        self._embedder = None
        self._cross_encoder = None
        self._all_chunks = None
        self._bm25_index = None
        self._client = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = EmbeddingManager()
        return self._embedder

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            self._cross_encoder = _get_shared_cross_encoder()
        return self._cross_encoder

    @property
    def all_chunks(self):
        if self._all_chunks is None:
            self._all_chunks = get_all_chunks_from_db()
        return self._all_chunks

    @property
    def bm25_index(self):
        if self._bm25_index is None:
            self._bm25_index = build_bm25_index(self.all_chunks)
        return self._bm25_index

    @property
    def client(self):
        if self._client is None:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
        return self._client

    def run(self, query_text):
        if not self.all_chunks:
            return "No documents found in the database. Please add documents first."

        vector_search = query_collection(query_text, self.embedder, 3) # gives the format of the chroma db
        vector_chunks = [
            {"document": doc, "metadata": meta}
            for doc, meta in zip(vector_search["documents"][0], vector_search["metadatas"][0])
        ]

        bm25_results_keyword_search = bm25_search(query_text, self.bm25_index, self.all_chunks, 10)

        final_results = merge_rrf(vector_chunks, bm25_results_keyword_search)

        reranked_results = rerank(query_text, final_results, self.cross_encoder, top_n=4)

        print(f"This are the {reranked_results}")

        prompt = build_prompt(query_text, reranked_results)

        models_to_try = ["gemini-3.6-flash", "gemini-3.5-flash"]
        last_err = None
        for model_name in models_to_try:
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    return response.text # send the actual required resp.
                except Exception as e:
                    last_err = e
                    if "503" in str(e):
                        time.sleep(1)
                        continue
                    raise e
        raise last_err

if __name__ == "__main__":
    rag = RAGPipeline()
    result = rag.run("what is the para virtualisation?")
    print("\n--- Final Generated Answer ---")
    print(result)