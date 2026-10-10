from typing import Optional
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


def rank_jds(
    resume_text: str,
    resume_id: str,
    top_n: Optional[int] = 5,
    preferred_location: Optional[str] = None,
    preferred_arrangement: Optional[str] = None
) -> RankingResult:
    """
    Tier 1 Ranking:
    Embeds resume_text once, computes cosine similarity against all stored JDs,
    applies deterministic preference boosts if specified,
    and returns top N (or all) ranked cards.
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

    resume_arr = np.array(resume_embedding, dtype=np.float32)
    norm_resume = float(np.linalg.norm(resume_arr))
    if norm_resume == 0.0:
        return RankingResult(resume_id=resume_id, results=[])

    scored: list[tuple[str, float]] = []
    # Vectorized cosine computation across stored JD vectors
    jd_ids = [j_id for j_id, _ in all_jds]
    jd_matrix = np.array([j_vec for _, j_vec in all_jds], dtype=np.float32)
    dots = np.dot(jd_matrix, resume_arr)
    norms_jd = np.linalg.norm(jd_matrix, axis=1)

    for i, j_id in enumerate(jd_ids):
        n_jd = float(norms_jd[i])
        if n_jd == 0.0:
            sim = 0.0
        else:
            sim = float(dots[i] / (n_jd * norm_resume))

        # Deterministic preference boost
        boost = 0.0
        rec = jd_index.get_jd_record(j_id)
        if rec and rec.location:
            jd_loc_lower = rec.location.lower()
            if preferred_location and preferred_location.strip():
                pref_loc_clean = preferred_location.strip().lower()
                if pref_loc_clean in jd_loc_lower:
                    boost += 0.05
            if preferred_arrangement and preferred_arrangement.strip():
                pref_arr_clean = preferred_arrangement.strip().lower()
                if pref_arr_clean != "any":
                    if pref_arr_clean in ["on-site", "onsite"]:
                        if any(term in jd_loc_lower for term in ["on-site", "onsite", "in-person", "office"]):
                            boost += 0.03
                    elif pref_arr_clean in jd_loc_lower:
                        boost += 0.03

        boosted_sim = sim + boost
        scored.append((j_id, boosted_sim))

    # Sort descending by cosine similarity, with deterministic jd_id tie-breaker
    scored.sort(key=lambda x: (-x[1], str(x[0])))
    if top_n is not None and top_n > 0:
        selected_scored = scored[:top_n]
    else:
        selected_scored = scored

    cards: list[RankingCard] = []
    for rank_idx, (j_id, sim) in enumerate(selected_scored, start=1):
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
