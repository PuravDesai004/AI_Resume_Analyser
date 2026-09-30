import os
import csv
import json
import re
from typing import Optional

from rapidfuzz import process, fuzz


from collections import defaultdict
from functools import lru_cache

# In-memory singleton cache to prevent redundant CSV reads across pipeline/test instances
_TAXONOMY_CACHE: dict[tuple[str, str], dict] = {}


def normalize_label(label: str) -> str:
    """Normalize label: lowercase, strip, and collapse whitespace."""
    if not label:
        return ""
    return " ".join(label.lower().split())


class SkillIndex:
    """Unified lookup index spanning ESCO and custom skills taxonomy."""

    def __init__(
        self,
        esco_csv_path: str = "data/skills_en.csv",
        custom_json_path: str = "data/custom_skills.json"
    ):
        if not os.path.exists(esco_csv_path):
            raise FileNotFoundError(
                f"ESCO taxonomy file not found at expected path: '{esco_csv_path}'. "
                f"Please download skills_en.csv from https://esco.ec.europa.eu/en/use-esco/download"
            )
        if not os.path.exists(custom_json_path):
            raise FileNotFoundError(
                f"Custom skills file not found at expected path: '{custom_json_path}'."
            )

        self.esco_csv_path = esco_csv_path
        self.custom_json_path = custom_json_path

        # Memoization cache for repeat fuzzy matches
        self._fuzzy_cache: dict[tuple[str, int], Optional[dict]] = {}
        self._exact_cache: dict[str, Optional[dict]] = {}

        # Check singleton taxonomy cache
        cache_key = (os.path.abspath(esco_csv_path), os.path.abspath(custom_json_path))
        if cache_key in _TAXONOMY_CACHE:
            cached = _TAXONOMY_CACHE[cache_key]
            self.by_label = cached["by_label"]
            self.all_labels = cached["all_labels"]
            self._labels_by_len = cached["_labels_by_len"]
            self._source_counts = cached["_source_counts"]
            return

        self.by_label: dict[str, dict] = {}
        self.all_labels: list[str] = []
        self._labels_by_len: dict[int, list[str]] = defaultdict(list)
        self._source_counts: dict[str, int] = {"esco": 0, "custom": 0}

        self._load_sources()

        # Cache for instant subsequent initializations
        _TAXONOMY_CACHE[cache_key] = {
            "by_label": self.by_label,
            "all_labels": self.all_labels,
            "_labels_by_len": self._labels_by_len,
            "_source_counts": self._source_counts
        }

    def _load_sources(self) -> None:
        # 1. Load ESCO dataset
        with open(self.esco_csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uri = row.get("conceptUri", "").strip()
                pref_label = row.get("preferredLabel", "").strip()
                alt_labels_raw = row.get("altLabels", "").strip()

                if not pref_label:
                    continue

                record = {
                    "skill_id": uri,
                    "preferred_label": pref_label,
                    "source": "esco"
                }

                norm_pref = normalize_label(pref_label)
                if norm_pref not in self.by_label:
                    self.by_label[norm_pref] = record
                    self._source_counts["esco"] += 1

                if alt_labels_raw:
                    for alt in alt_labels_raw.split("\n"):
                        cleaned = alt.strip()
                        if cleaned:
                            norm_alt = normalize_label(cleaned)
                            if norm_alt not in self.by_label:
                                self.by_label[norm_alt] = record
                                self._source_counts["esco"] += 1

        # 2. Load custom skills dataset
        with open(self.custom_json_path, mode="r", encoding="utf-8") as f:
            custom_data = json.load(f)

        for item in custom_data:
            skill_id = item["id"]
            pref_label = item["preferred_label"]
            alt_labels = item.get("alt_labels", [])

            record = {
                "skill_id": skill_id,
                "preferred_label": pref_label,
                "source": "custom"
            }

            all_custom_labels = [pref_label] + alt_labels
            for label in all_custom_labels:
                norm_lbl = normalize_label(label)
                if norm_lbl and norm_lbl not in self.by_label:
                    self.by_label[norm_lbl] = record
                    self._source_counts["custom"] += 1

        self.all_labels = list(self.by_label.keys())
        # Index labels by length for ultra-fast candidate pre-filtering
        for lbl in self.all_labels:
            self._labels_by_len[len(lbl)].append(lbl)

    def exact_match(self, phrase: str) -> Optional[dict]:
        """Perform exact normalized dictionary lookup in O(1) with memoization."""
        if not phrase:
            return None
        if phrase in self._exact_cache:
            return self._exact_cache[phrase]
        norm = normalize_label(phrase)
        res = self.by_label.get(norm)
        self._exact_cache[phrase] = res
        return res

    def fuzzy_match(self, phrase: str, threshold: int = 85) -> Optional[dict]:
        """
        Perform fuzzy match using RapidFuzz against indexed labels.
        Uses length-bucketed candidate pre-filtering and memoization.
        Returns matched record and score if meeting threshold.
        """
        if not phrase or not self.all_labels:
            return None

        cache_key = (phrase, threshold)
        if cache_key in self._fuzzy_cache:
            return self._fuzzy_cache[cache_key]

        norm = normalize_label(phrase)
        norm_len = len(norm)
        if norm_len == 0:
            self._fuzzy_cache[cache_key] = None
            return None

        # Pre-filter candidate labels using pre-indexed length buckets
        # Condition: min(len, norm_len) / max(len, norm_len) >= 0.5
        # => ceil(norm_len * 0.5) <= len <= floor(norm_len * 2.0)
        min_k = max(1, (norm_len + 1) // 2)
        max_k = norm_len * 2

        candidates = []
        for k in range(min_k, max_k + 1):
            if k < 3 and k != norm_len:
                continue
            bucket = self._labels_by_len.get(k)
            if bucket:
                candidates.extend(bucket)

        if not candidates:
            self._fuzzy_cache[cache_key] = None
            return None

        match_result = process.extractOne(
            norm,
            candidates,
            scorer=fuzz.QRatio,
            score_cutoff=threshold
        )

        res = None
        if match_result:
            matched_label, score, _ = match_result
            record = self.by_label.get(matched_label)
            if record:
                res = {
                    "skill_id": record["skill_id"],
                    "preferred_label": record["preferred_label"],
                    "source": record["source"],
                    "matched_label": matched_label,
                    "score": float(score)
                }

        self._fuzzy_cache[cache_key] = res
        return res

    def stats(self) -> dict:
        """Counts of indexed labels by source."""
        return {
            "esco": self._source_counts["esco"],
            "custom": self._source_counts["custom"],
            "total_labels": len(self.by_label)
        }


if __name__ == "__main__":
    index = SkillIndex()
    print("SkillIndex stats:", index.stats())
    fastapi_match = index.exact_match("FastAPI")
    print("FastAPI match:", fastapi_match)
    assert fastapi_match is not None and fastapi_match["source"] == "custom"

    python_match = index.exact_match("Python")
    print("Python match:", python_match)
    assert python_match is not None and python_match["source"] == "esco"

    python_lower_match = index.exact_match("python")
    assert python_lower_match == python_match
    print("esco_loader acceptance criteria passed!")
