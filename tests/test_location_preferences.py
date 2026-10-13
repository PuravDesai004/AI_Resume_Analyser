"""Offline normalization, real cities, hard distance cutoffs and ranking bonuses."""
import math
import os
import unittest
from unittest.mock import MagicMock, patch

from location_preferences import (bonus_for_distance, distance_km, location_preference,
                                  normalize_name, resolve_city)
from schemas import JDRecord
import ranking
from decision_ranking import Assessment, DecisionOutcome


class CityLocationTests(unittest.TestCase):
    def test_case_spaces_accents_aliases_and_work_annotations(self):
        mumbai = resolve_city("Mumbai")
        for name in ("  MUMBAI ", "Bombay", "Mumbai, Maharashtra, India", "Mumbai (Hybrid)"):
            self.assertEqual(resolve_city(name).identifier, mumbai.identifier)
        self.assertEqual(normalize_name(" São   Paulo "), "sao paulo")
        self.assertEqual(resolve_city("São Paulo, Brazil").name, "São Paulo")
        self.assertEqual(resolve_city("Poona").identifier, resolve_city("Pune").identifier)
        self.assertEqual(resolve_city("Remote, New York").identifier, resolve_city("New York").identifier)

    def test_city_state_country_qualifiers_and_ambiguous_locations(self):
        self.assertEqual(resolve_city("London, UK").country, "GB")
        self.assertEqual(resolve_city("London, Ontario, Canada").country, "CA")
        self.assertEqual(resolve_city("Paris, TX").region, "TX")
        self.assertEqual(resolve_city("San Francisco, CA (Remote)").region, "CA")
        self.assertIsNone(resolve_city("Mumbai, USA"))
        self.assertIsNone(resolve_city("Springfield"))
        self.assertIsNone(resolve_city("Mumbai / Pune"))
        self.assertIsNone(resolve_city("Not-a-real-city"))
        self.assertIsNone(resolve_city("Remote"))
        self.assertIsNone(resolve_city("India"))
        self.assertEqual(resolve_city("Singapore").country, "SG")

    def test_user_hierarchy_inclusive_boundaries_and_no_farther_nearest_bonus(self):
        for distance, bonus in ((0, 4), (99.99, 4), (100, 4), (100.001, 3),
                                (200, 3), (200.001, 2), (300, 2), (300.001, 0), (10000, 0)):
            self.assertEqual(bonus_for_distance(distance), bonus)
        for invalid in (-1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                bonus_for_distance(invalid)

    def test_mumbai_examples_use_real_coordinates_and_exact_city_identity(self):
        expected = {"Mumbai": 5, "Bombay": 5, "Navi Mumbai": 4, "Pune": 3, "Ratnagiri": 2, "Ahmedabad": 0}
        for city, bonus in expected.items():
            self.assertEqual(location_preference("Mumbai", city).bonus, bonus)
        pune = location_preference("Mumbai", "Pune")
        self.assertAlmostEqual(pune.distance_km, 119.5, delta=1)
        ahmedabad = location_preference("Mumbai", "Ahmedabad")
        self.assertGreater(ahmedabad.distance_km, 300)
        self.assertIn("Beyond 300", ahmedabad.reason)
        self.assertEqual(location_preference("Mumbai", "Bombay").distance_km, 0)

    def test_no_guess_for_unknown_preference_job_or_missing_city(self):
        for preferred, job in ((None, "Mumbai"), ("Unknownville", "Mumbai"),
                               ("Mumbai", "Unknownville"), ("Mumbai", "Remote"),
                               ("Mumbai", ""), ("Mumbai", "Pune / Ahmedabad")):
            result = location_preference(preferred, job)
            self.assertEqual(result.bonus, 0)
            self.assertIsNone(result.distance_km)

    def test_haversine_symmetry(self):
        a, b = resolve_city("Mumbai"), resolve_city("Pune")
        self.assertEqual(distance_km(a, a), 0)
        self.assertAlmostEqual(distance_km(a, b), distance_km(b, a))


class LocationRankingTests(unittest.TestCase):
    def setUp(self):
        self.embedder = MagicMock()
        self.embedder.embed_chunk.return_value = [1, 0]
        ranking._embed_resume.cache_clear()
        self.snapshot = [(JDRecord(jd_id=city, title=city, company="Example", location=city,
                                   snippet="Python", full_text="Python"), [1, 0])
                         for city in ["Mumbai", "Navi Mumbai", "Pune", "Nashik", "Ahmedabad"]]

    def tearDown(self):
        ranking._embed_resume.cache_clear()

    def test_bonus_hierarchy_affects_shortlist_and_is_separate_from_similarity(self):
        with patch.object(ranking, "_get_embedder", return_value=self.embedder), \
                patch.object(ranking.jd_index, "get_ranking_snapshot", return_value=self.snapshot), \
                patch.dict(os.environ, {"RANK_DECISION_ENABLED": "false"}):
            result = ranking.rank_jds("Resume", "r", preferred_location="Mumbai")
        bonuses = {c.jd_id: c.location_bonus for c in result.results}
        self.assertEqual(bonuses, {"Mumbai": 5, "Navi Mumbai": 4, "Pune": 3, "Nashik": 3, "Ahmedabad": 0})
        self.assertTrue(all(c.embedding_similarity == 100 for c in result.results))
        self.assertTrue(all(c.score_source == "similarity" for c in result.results))
        self.assertEqual(result.results[0].jd_id, "Mumbai")
        self.assertEqual(result.results[-1].jd_id, "Ahmedabad")

    def test_mercury_fit_is_unchanged_and_location_bonus_changes_order(self):
        assessments = {"Mumbai": Assessment(70, "Moderate", .9), "Pune": Assessment(73, "Moderate", .9)}
        outcome = DecisionOutcome(assessments, "applied", "Applied", "test-model")
        selected = [pair for pair in self.snapshot if pair[0].jd_id in assessments]
        with patch.object(ranking, "_get_embedder", return_value=self.embedder), \
                patch.object(ranking.jd_index, "get_ranking_snapshot", return_value=selected), \
                patch.object(ranking, "assess_shortlist", return_value=outcome):
            result = ranking.rank_jds("Resume", "r", preferred_location="Mumbai")
        self.assertEqual([c.jd_id for c in result.results], ["Pune", "Mumbai"])
        self.assertEqual([c.model_fit_score for c in result.results], [73, 70])
        self.assertEqual([c.ranking_score for c in result.results], [76, 75])
        self.assertEqual([c.location_bonus for c in result.results], [3, 5])
        self.assertTrue(all(c.score_source == "mercury" for c in result.results))


if __name__ == "__main__":
    unittest.main()
