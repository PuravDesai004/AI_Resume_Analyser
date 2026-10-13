from typing import Optional
from functools import lru_cache
import numpy as np

import jd_index
from embedding_manager import EmbeddingManager
from schemas import RankingCard, RankingResult
from decision_ranking import assess_shortlist, SHORTLIST_LIMIT
from location_preferences import location_preference, resolve_city

_embedder = None


def _get_embedder() -> EmbeddingManager:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingManager()
    return _embedder


@lru_cache(maxsize=32)
def _embed_resume(embedder, text):
    return tuple(embedder.embed_chunk(text))


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
    Mercury optionally reranks the resulting shortlist using full document pairs.
    """
    clean_resume = resume_text.strip()
    if not clean_resume:
        return RankingResult(resume_id=resume_id, results=[])

    all_jds = jd_index.get_ranking_snapshot()
    if not all_jds:
        return RankingResult(resume_id=resume_id, results=[])

    embedder = _get_embedder()
    resume_embedding = _embed_resume(embedder, clean_resume)

    resume_arr = np.array(resume_embedding, dtype=np.float32)
    norm_resume = float(np.linalg.norm(resume_arr))
    if norm_resume == 0.0:
        return RankingResult(resume_id=resume_id, results=[])

    scored: list[tuple[str, float]] = []
    raw_scores = {}
    boosts = {}
    locations = {}
    arrangements = {}
    # Vectorized cosine computation across stored JD vectors
    records = {record.jd_id: record for record, _ in all_jds}
    jd_ids = list(records)
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
        rec = records[j_id]
        location = location_preference(preferred_location, rec.location)
        locations[j_id] = location
        arrangement_points = 0
        if rec and rec.location:
            jd_loc_lower = rec.location.lower()
            if preferred_arrangement and preferred_arrangement.strip():
                pref_arr_clean = preferred_arrangement.strip().lower()
                if pref_arr_clean != "any":
                    if pref_arr_clean in ["on-site", "onsite"]:
                        if any(term in jd_loc_lower for term in ["on-site", "onsite", "in-person", "office"]):
                            arrangement_points = 3
                    elif pref_arr_clean in jd_loc_lower:
                        arrangement_points = 3

        arrangements[j_id] = arrangement_points
        boost = (location.bonus + arrangement_points) / 100
        boosted_sim = sim + boost
        raw_scores[j_id] = sim
        boosts[j_id] = boost
        scored.append((j_id, boosted_sim))

    # Sort descending by cosine similarity, with deterministic jd_id tie-breaker
    scored.sort(key=lambda x: (-x[1], str(x[0])))
    # The number displayed is independent of the AI budget. Always assess at
    # most the leading 25; "All jobs" never expands this provider shortlist.
    baseline_ranks = {job_id: index for index, (job_id, _) in enumerate(scored, 1)}
    shortlist = scored[:SHORTLIST_LIMIT]
    shortlist_ids = {job_id for job_id, _ in shortlist}
    outcome = assess_shortlist(clean_resume, [records[job_id] for job_id, _ in shortlist])
    selected_scored = scored
    if outcome.status == "applied":
        selected_scored = sorted(shortlist, key=lambda pair: (
            -(outcome.assessments[pair[0]].score + 100 * boosts[pair[0]]),
            -pair[1], str(pair[0]),
        )) + scored[SHORTLIST_LIMIT:]
    if top_n is not None and top_n > 0:
        selected_scored = selected_scored[:top_n]

    cards: list[RankingCard] = []
    for rank_idx, (j_id, sim) in enumerate(selected_scored, start=1):
        rec = records[j_id]
        if rec is None:
            continue

        score_int = max(0, min(100, round(sim * 100)))
        assessed = outcome.status == "applied" and j_id in outcome.assessments
        cards.append(
            RankingCard(
                jd_id=rec.jd_id,
                rank=rank_idx,
                similarity_score=score_int,
                title=rec.title,
                company=rec.company,
                location=rec.location,
                snippet=rec.snippet,
                embedding_similarity=max(0, min(100, round(raw_scores[j_id] * 100))),
                preference_boost=round(boosts[j_id] * 100),
                location_bonus=locations[j_id].bonus,
                arrangement_bonus=arrangements[j_id],
                location_distance_km=locations[j_id].distance_km,
                location_reason=locations[j_id].reason,
                score_source="mercury" if assessed else "similarity",
                fit_status="assessed" if assessed else "not_reviewed" if j_id not in shortlist_ids else "fallback",
                ranking_score=round(outcome.assessments[j_id].score + 100 * boosts[j_id], 3) if assessed else round(max(0, min(108, sim * 100)), 3),
                baseline_rank=baseline_ranks[j_id],
                model_fit=(outcome.assessments[j_id].label if assessed else "Unclear") if j_id in outcome.assessments else None,
                model_fit_score=outcome.assessments[j_id].score if j_id in outcome.assessments else None,
                decision_confidence=outcome.assessments[j_id].confidence if j_id in outcome.assessments else None,
            )
        )

    note = outcome.note
    if len(scored) > SHORTLIST_LIMIT:
        note += f" AI review is limited to the top {SHORTLIST_LIMIT} embedding matches. Remaining jobs follow in similarity order."
    if preferred_location and preferred_location.strip() and resolve_city(preferred_location) is None:
        note += " Location preference could not be resolved; enter a city with country/state. No location bonuses applied."
    return RankingResult(resume_id=resume_id, results=cards,
                         ranking_method="mercury" if outcome.status == "applied" else "similarity",
                         decision_status=outcome.status, ranking_note=note,
                         decision_model=outcome.model, decision_shortlist_size=len(shortlist), total_jobs=len(scored))


if __name__ == "__main__":
    print("Testing ranking module...")
    empty_result = rank_jds("Python developer resume", "res_001")
    print("Ranking with 0 JDs:", empty_result.model_dump())
