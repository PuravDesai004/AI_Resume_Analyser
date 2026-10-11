# Connected RAG engine

Adapted from https://github.com/shahmit107/RAG.git at source commit
`f598c4ecb16dc930addb31e7d48da33d3581d31d`. The reusable source was under
`backend/pipeline/`; that project's `docs/` folder holds screenshots.

The token chunker, Gemini embedding wrapper, BM25 retrieval, reciprocal rank
fusion, CrossEncoder reranking, grounded prompt design, and Gemini/Groq fallback
are adapted here. `chat_service.py` adds immutable document-context snapshots,
scoped Chroma retrieval, validated citations, and conversation history. Local
copies make the connection portable without requiring the other project's server.

Changes include retrieval task types, required session/context filters, merging
by chunk ID, lazy optional reranking, typed generation, and removal of document
logging. Runtime storage uses the analyser's local database.
