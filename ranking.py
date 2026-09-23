import numpy as np

import jd_index
from embedding_manager import EmbeddingManager
from schemas import RankingCard, RankingResult

_embedder = None


def _get_embedder() -> EmbeddingManager:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingManager()
    return _embedder


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def rank_jds(resume_text: str, resume_id: str) -> RankingResult:
    """
    Tier 1 Ranking:
    Embeds resume_text once, computes cosine similarity against all stored JDs,
    and returns top 5 (or fewer) ranked cards.
    Zero Gemini / LLM calls.
    """
    clean_resume = resume_text.strip()
    if not clean_resume:
        return RankingResult(resume_id=resume_id, results=[])

    all_jds = jd_index.get_all_jd_embeddings()
    if not all_jds:
        return RankingResult(resume_id=resume_id, results=[])

    embedder = _get_embedder()
    resume_embedding = embedder.embed_chunk(clean_resume)

    scored: list[tuple[str, float]] = []
    for j_id, j_vec in all_jds:
        sim = _cosine_similarity(resume_embedding, j_vec)
        scored.append((j_id, sim))

    # Sort descending by cosine similarity, with deterministic jd_id tie-breaker
    scored.sort(key=lambda x: (-x[1], str(x[0])))
    top_5 = scored[:5]

    cards: list[RankingCard] = []
    for rank_idx, (j_id, sim) in enumerate(top_5, start=1):
        rec = jd_index.get_jd_record(j_id)
        if rec is None:
            continue

        score_int = max(0, min(100, round(sim * 100)))
        cards.append(
            RankingCard(
                jd_id=rec.jd_id,
                rank=rank_idx,
                similarity_score=score_int,
                title=rec.title,
                company=rec.company,
                location=rec.location,
                snippet=rec.snippet
            )
        )

    return RankingResult(resume_id=resume_id, results=cards)


if __name__ == "__main__":
    print("Testing ranking module...")
    empty_result = rank_jds("Python developer resume", "res_001")
    print("Ranking with 0 JDs:", empty_result.model_dump())
