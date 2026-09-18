import os
import csv
from typing import Optional

try:
    from rapidfuzz import process, fuzz
except ImportError:
    process = None
    fuzz = None


def normalize_label(label: str) -> str:
    """Normalize label for consistent dictionary lookup."""
    return label.strip().lower()


def load_esco_skills(csv_path: str = "data/skills_en.csv") -> dict:
    """
    Reads the ESCO skills CSV and constructs lookup indexes.
    Expects columns: conceptUri, preferredLabel, altLabels, description, skillType.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"ESCO taxonomy file not found at '{csv_path}'. "
            f"Please download 'skills_en.csv' from https://esco.ec.europa.eu/en/use-esco/download "
            f"and place it in the 'data/' directory."
        )

    by_label: dict[str, dict] = {}
    all_labels_set: set[str] = set()

    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uri = row.get("conceptUri", "").strip()
            pref_label = row.get("preferredLabel", "").strip()
            alt_labels_raw = row.get("altLabels", "").strip()

            if not pref_label:
                continue

            record = {
                "uri": uri,
                "preferred_label": pref_label
            }

            norm_pref = normalize_label(pref_label)
            by_label[norm_pref] = record
            all_labels_set.add(norm_pref)

            # Process alt labels (often separated by newlines or pipes in ESCO exports)
            if alt_labels_raw:
                delimiters = ["\n", "|", "\r\n"]
                alt_list = [alt_labels_raw]
                for d in delimiters:
                    expanded = []
                    for item in alt_list:
                        expanded.extend(item.split(d))
                    alt_list = expanded

                for alt in alt_list:
                    cleaned_alt = alt.strip()
                    if cleaned_alt:
                        norm_alt = normalize_label(cleaned_alt)
                        if norm_alt not in by_label:
                            by_label[norm_alt] = record
                        all_labels_set.add(norm_alt)

    return {
        "by_label": by_label,
        "all_labels": list(all_labels_set)
    }


class ESCOIndex:
    """Singleton-style in-memory lookup index for ESCO skills."""

    def __init__(self, csv_path: str = "data/skills_en.csv"):
        data = load_esco_skills(csv_path)
        self.by_label: dict[str, dict] = data["by_label"]
        self.all_labels: list[str] = data["all_labels"]

    def exact_match(self, phrase: str) -> Optional[dict]:
        """Perform exact normalized dictionary lookup."""
        normalized = normalize_label(phrase)
        return self.by_label.get(normalized)

    def fuzzy_match(self, phrase: str, threshold: int = 80) -> Optional[dict]:
        """
        Perform fuzzy match using rapidfuzz against all known ESCO labels.
        Returns matched record plus match score if above threshold.
        """
        if not process or not self.all_labels:
            return None

        normalized = normalize_label(phrase)
        candidates = [
            label for label in self.all_labels
            if min(len(label), len(normalized)) / max(len(label), len(normalized)) >= 0.5
            and (len(label) >= 3 or len(label) == len(normalized))
        ]
        match_result = process.extractOne(
            normalized,
            candidates,
            scorer=fuzz.WRatio,
            score_cutoff=threshold
        )

        if match_result:
            matched_label, score, _ = match_result
            record = self.by_label.get(matched_label)
            if record:
                return {
                    "uri": record["uri"],
                    "preferred_label": record["preferred_label"],
                    "matched_label": matched_label,
                    "score": float(score)
                }

        return None

    def get_all_labels(self) -> list[str]:
        """Return list of all registered skill labels."""
        return self.all_labels


if __name__ == "__main__":
    test_path = "data/skills_en.csv"
    if os.path.exists(test_path):
        index = ESCOIndex(test_path)
        print(f"Loaded {len(index.all_labels)} labels.")
        print("Exact match 'python':", index.exact_match("python"))
        print("Fuzzy match 'React.js':", index.fuzzy_match("React.js"))
    else:
        print(f"Notice: '{test_path}' not found yet. Place the ESCO CSV in 'data/' to test.")
