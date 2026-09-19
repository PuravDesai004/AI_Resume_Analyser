import os
import csv
import json
import re
from typing import Optional

from rapidfuzz import process, fuzz


def normalize_label(label: str) -> str:
    """Normalize label: lowercase, strip, and collapse whitespace."""
    if not label:
        return ""
    return re.sub(r"\s+", " ", label.strip().lower())


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

        self.by_label: dict[str, dict] = {}
        self.all_labels: list[str] = []
        self._source_counts: dict[str, int] = {"esco": 0, "custom": 0}

        self._load_sources()

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

    def exact_match(self, phrase: str) -> Optional[dict]:
        """Perform exact normalized dictionary lookup in O(1)."""
        if not phrase:
            return None
        norm = normalize_label(phrase)
        return self.by_label.get(norm)

    def fuzzy_match(self, phrase: str, threshold: int = 85) -> Optional[dict]:
        """
        Perform fuzzy match using RapidFuzz against indexed labels.
        Returns matched record and score if meeting threshold.
        """
        if not phrase or not self.all_labels:
            return None

        norm = normalize_label(phrase)

        # Pre-filter candidate labels to keep fuzzy match fast and accurate
        norm_len = len(norm)
        candidates = [
            lbl for lbl in self.all_labels
            if min(len(lbl), norm_len) / max(len(lbl), norm_len) >= 0.5
            and (len(lbl) >= 3 or len(lbl) == norm_len)
        ]

        if not candidates:
            return None

        match_result = process.extractOne(
            norm,
            candidates,
            scorer=fuzz.QRatio,
            score_cutoff=threshold
        )

        if match_result:
            matched_label, score, _ = match_result
            record = self.by_label.get(matched_label)
            if record:
                return {
                    "skill_id": record["skill_id"],
                    "preferred_label": record["preferred_label"],
                    "source": record["source"],
                    "matched_label": matched_label,
                    "score": float(score)
                }

        return None

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
