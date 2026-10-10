import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from server import app
import jd_index
import ranking
from analysis_pipeline import AnalysisPipeline


class TestNewFeatures(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_feature_1_top_n_selector_ranking(self):
        """Test top_n parameter in ranking.py, AnalysisPipeline, and /rank endpoint."""
        jd_index.clear_store_for_testing()

        fake_vec = [0.1] * 3072
        with patch.object(jd_index, "_get_embedder") as mock_emb_fn:
            mock_emb = MagicMock()
            mock_emb.embed_chunk.return_value = fake_vec
            mock_emb_fn.return_value = mock_emb

            # Ingest 7 JDs
            created_ids = []
            for i in range(7):
                rec = jd_index.add_jd(f"Role {i}", f"Company {i}", f"Location {i}", f"Job text {i} " * 10)
                created_ids.append(rec.jd_id)

        with patch.object(ranking, "_get_embedder") as mock_rank_emb_fn:
            mock_rank_emb = MagicMock()
            mock_rank_emb.embed_chunk.return_value = fake_vec
            mock_rank_emb_fn.return_value = mock_rank_emb

            # Test default top_n=5
            res_default = ranking.rank_jds("Resume text " * 10, "res_test")
            self.assertEqual(len(res_default.results), 5)
            self.assertEqual([c.rank for c in res_default.results], [1, 2, 3, 4, 5])

            # Test top_n=2
            res_2 = ranking.rank_jds("Resume text " * 10, "res_test", top_n=2)
            self.assertEqual(len(res_2.results), 2)
            self.assertEqual([c.rank for c in res_2.results], [1, 2])

            # Test top_n=None (All)
            res_all = ranking.rank_jds("Resume text " * 10, "res_test", top_n=None)
            self.assertEqual(len(res_all.results), 7)
            self.assertEqual([c.rank for c in res_all.results], list(range(1, 8)))

            # Test top_n=0 (All)
            res_zero = ranking.rank_jds("Resume text " * 10, "res_test", top_n=0)
            self.assertEqual(len(res_zero.results), 7)

            # Test top_n=10 with only 7 JDs (no crash, returns 7)
            res_10 = ranking.rank_jds("Resume text " * 10, "res_test", top_n=10)
            self.assertEqual(len(res_10.results), 7)

            # Test API /rank with top_n=3
            api_resp_3 = self.client.post("/rank", json={
                "resume_text": "Resume text " * 10,
                "resume_id": "res_test",
                "top_n": 3
            })
            self.assertEqual(api_resp_3.status_code, 200)
            self.assertEqual(len(api_resp_3.json()["results"]), 3)

            # Test API /rank with top_n=None
            api_resp_all = self.client.post("/rank", json={
                "resume_text": "Resume text " * 10,
                "resume_id": "res_test",
                "top_n": None
            })
            self.assertEqual(api_resp_all.status_code, 200)
            self.assertEqual(len(api_resp_all.json()["results"]), 7)

    def test_feature_6_preferences_boost(self):
        """Test additive preference boost and clamping in ranking."""
        jd_index.clear_store_for_testing()

        # Let vec_a and vec_b have unit norm:
        # e.g., resume vector [1, 0, ...]
        # JD 1 vector: [0.72, sqrt(1 - 0.72^2), ...] -> cosine similarity = 0.72
        # JD 2 vector: [0.99, sqrt(1 - 0.99^2), ...] -> cosine similarity = 0.99 (test clamp)
        import math
        vec_resume = [1.0] + [0.0] * 3071
        vec_jd1 = [0.72, math.sqrt(1 - 0.72**2)] + [0.0] * 3070
        vec_jd2 = [0.99, math.sqrt(1 - 0.99**2)] + [0.0] * 3070

        with patch.object(jd_index, "_get_embedder") as mock_emb_fn:
            mock_emb = MagicMock()
            mock_emb.embed_chunk.side_effect = [vec_jd1, vec_jd2]
            mock_emb_fn.return_value = mock_emb

            rec1 = jd_index.add_jd(
                "Software Engineer", "Acme", "Remote, New York", "Some text " * 10
            )
            rec2 = jd_index.add_jd(
                "Senior Engineer", "Beta", "London, On-site", "Some text " * 10
            )

        with patch.object(ranking, "_get_embedder") as mock_rank_emb_fn:
            mock_rank_emb = MagicMock()
            mock_rank_emb.embed_chunk.return_value = vec_resume
            mock_rank_emb_fn.return_value = mock_rank_emb

            # 1. No preferences: raw similarity
            # rec1: 0.72 -> 72
            # rec2: 0.99 -> 99
            res_no_pref = ranking.rank_jds("Resume text", "res_001")
            scores_no_pref = {c.jd_id: c.similarity_score for c in res_no_pref.results}
            self.assertEqual(scores_no_pref[rec1.jd_id], 72)
            self.assertEqual(scores_no_pref[rec2.jd_id], 99)

            # 2. Location preference: "New York" -> +0.05
            # rec1 gets 0.72 + 0.05 = 0.77 -> 77
            res_loc = ranking.rank_jds(
                "Resume text", "res_001", preferred_location="New York"
            )
            scores_loc = {c.jd_id: c.similarity_score for c in res_loc.results}
            self.assertEqual(scores_loc[rec1.jd_id], 77)
            self.assertEqual(scores_loc[rec2.jd_id], 99)

            # 3. Work arrangement preference: "Remote" -> +0.03
            # rec1 gets 0.72 + 0.03 = 0.75 -> 75
            res_arr = ranking.rank_jds(
                "Resume text", "res_001", preferred_arrangement="Remote"
            )
            scores_arr = {c.jd_id: c.similarity_score for c in res_arr.results}
            self.assertEqual(scores_arr[rec1.jd_id], 75)
            self.assertEqual(scores_arr[rec2.jd_id], 99)

            # 4. Both preferences: "New York" (+0.05) and "Remote" (+0.03) -> +0.08
            # rec1 gets 0.72 + 0.05 + 0.03 = 0.80 -> 80
            res_both = ranking.rank_jds(
                "Resume text",
                "res_001",
                preferred_location="New York",
                preferred_arrangement="Remote"
            )
            scores_both = {c.jd_id: c.similarity_score for c in res_both.results}
            self.assertEqual(scores_both[rec1.jd_id], 80)

            # 5. Clamp to 100: rec2 (0.99) with "London" (+0.05) and "On-site" (+0.03) -> 0.99 + 0.08 = 1.07 -> clamped to 100
            res_clamped = ranking.rank_jds(
                "Resume text",
                "res_001",
                preferred_location="London",
                preferred_arrangement="On-site"
            )
            scores_clamped = {c.jd_id: c.similarity_score for c in res_clamped.results}
            self.assertEqual(scores_clamped[rec2.jd_id], 100)

            # 6. API call with preferences
            api_resp = self.client.post("/rank", json={
                "resume_text": "Resume text",
                "resume_id": "res_001",
                "preferred_location": "New York",
                "preferred_arrangement": "Remote"
            })
            self.assertEqual(api_resp.status_code, 200)
            api_scores = {c["jd_id"]: c["similarity_score"] for c in api_resp.json()["results"]}
            self.assertEqual(api_scores[rec1.jd_id], 80)

    def test_feature_3_download_report(self):
        """Test markdown gap analysis report generation."""
        from report_generator import generate_markdown_report

        mock_analysis = {
            "resume_id": "res_100",
            "jd_id": "jd_200",
            "context_sufficient": True,
            "match_score": 88,
            "summary": "Candidate shows exceptional backend alignment.",
            "strengths": ["Deep Python expertise", "FastAPI architecture experience"],
            "weaknesses": ["Lacks production Kubernetes operations"],
            "skill_gap": {
                "matched": [{"skill_name": "Python", "match_type": "exact"}, {"skill_name": "FastAPI", "match_type": "exact"}],
                "missing_essential": ["Docker"],
                "missing_preferred": ["Kubernetes"],
                "extra": ["Flask"]
            },
            "requirement_match": [
                {"category": "Skills", "match_percent": 88},
                {"category": "Experience", "match_percent": 85}
            ],
            "recommendations": [
                {"skill": "Docker", "reason": "Required for microservice deployments"}
            ]
        }

        md = generate_markdown_report(mock_analysis, "Jane Doe", "Senior Python Engineer")

        # Verify key sections and formatting
        self.assertIn("# Gap Analysis Report: Jane Doe vs Senior Python Engineer", md)
        self.assertIn("## Match Score: 88%", md)
        self.assertIn("Candidate shows exceptional backend alignment.", md)
        self.assertIn("- Deep Python expertise", md)
        self.assertIn("- Lacks production Kubernetes operations", md)
        self.assertIn("| **Matched Skills** | Python, FastAPI |", md)
        self.assertIn("| **Missing Essential** | Docker |", md)
        self.assertIn("| **Missing Preferred** | Kubernetes |", md)
        self.assertIn("| **Candidate Extra** | Flask |", md)
        self.assertIn("| Skills | 88% |", md)
        self.assertIn("| Docker | Required for microservice deployments |", md)

    def test_feature_4_multi_jd_comparison(self):
        """Test Multi-JD Comparison chart generation and overlap metrics."""
        from app import make_multi_jd_grouped_bar

        analyses_map = {
            "jd_001": {
                "match_score": 85,
                "requirement_match": [
                    {"category": "Skills", "match_percent": 85},
                    {"category": "Experience", "match_percent": 80},
                    {"category": "Education", "match_percent": 100},
                    {"category": "Responsibilities", "match_percent": 75}
                ],
                "skill_gap": {
                    "matched": [{"skill_name": "Python"}, {"skill_name": "FastAPI"}],
                    "missing_essential": ["Docker", "Redis"],
                    "missing_preferred": ["Kubernetes"]
                }
            },
            "jd_002": {
                "match_score": 70,
                "requirement_match": [
                    {"category": "Skills", "match_percent": 70},
                    {"category": "Experience", "match_percent": 75},
                    {"category": "Education", "match_percent": 90},
                    {"category": "Responsibilities", "match_percent": 65}
                ],
                "skill_gap": {
                    "matched": [{"skill_name": "Python"}],
                    "missing_essential": ["Docker", "PostgreSQL"],
                    "missing_preferred": ["AWS"]
                }
            }
        }
        titles_map = {"jd_001": "Backend Dev", "jd_002": "Platform Dev"}

        fig = make_multi_jd_grouped_bar(analyses_map, titles_map)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 2)
        self.assertEqual(fig.data[0].name, "Backend Dev")
        self.assertEqual(fig.data[1].name, "Platform Dev")

        # Test shared missing essential skills computation
        set_1 = set(analyses_map["jd_001"]["skill_gap"]["missing_essential"])
        set_2 = set(analyses_map["jd_002"]["skill_gap"]["missing_essential"])
        shared = set_1.intersection(set_2)
        self.assertEqual(shared, {"Docker"})

    def test_feature_5_skill_gap_dashboard(self):
        """Test aggregation and chart rendering for Skill Gap Dashboard."""
        from app import aggregate_skill_gaps, make_skill_gap_dashboard_chart

        mock_analyses = [
            # Successful JD 1
            {
                "context_sufficient": True,
                "skill_gap": {
                    "matched": [{"skill_name": "Python"}, {"skill_name": "FastAPI"}],
                    "missing_essential": ["Docker", "Redis"],
                    "missing_preferred": ["Kubernetes"]
                }
            },
            # Successful JD 2
            {
                "context_sufficient": True,
                "skill_gap": {
                    "matched": [{"skill_name": "Python"}],
                    "missing_essential": ["Docker", "PostgreSQL"],
                    "missing_preferred": ["Kubernetes", "AWS"]
                }
            },
            # Failed / Insufficient JD 3 (should be skipped gracefully)
            {
                "context_sufficient": False,
                "insufficient_reason": "Text too short",
                "skill_gap": {}
            }
        ]

        ess_counts, pref_counts, match_counts, success_count = aggregate_skill_gaps(mock_analyses)

        self.assertEqual(success_count, 2)
        # Docker is missing in both JD 1 and JD 2
        self.assertEqual(ess_counts["Docker"], 2)
        self.assertEqual(ess_counts["Redis"], 1)
        self.assertEqual(ess_counts["PostgreSQL"], 1)
        # Kubernetes preferred missing in both
        self.assertEqual(pref_counts["Kubernetes"], 2)
        self.assertEqual(pref_counts["AWS"], 1)
        # Python matched in both
        self.assertEqual(match_counts["Python"], 2)
        self.assertEqual(match_counts["FastAPI"], 1)

        # Test chart with data
        fig_with_data = make_skill_gap_dashboard_chart(ess_counts, pref_counts)
        self.assertIsNotNone(fig_with_data)
        self.assertEqual(len(fig_with_data.data), 2)  # Essential and Preferred traces

        # Test chart with 0 missing skills
        fig_empty = make_skill_gap_dashboard_chart({}, {})
        self.assertIsNotNone(fig_empty)
        self.assertEqual(fig_empty.layout.title.text, "No skill gaps found across analyzed JDs")

    def test_markdown_report_sanitization_and_insufficient_context(self):
        """Test Markdown table sanitization (escaping pipes & newlines) and insufficient context handling."""
        from report_generator import _clean_table_cell, generate_markdown_report

        # 1. Test cell cleaner
        self.assertEqual(_clean_table_cell("Python | FastAPI"), r"Python \| FastAPI")
        self.assertEqual(_clean_table_cell("Line 1\nLine 2\r\nLine 3"), "Line 1 Line 2 Line 3")
        self.assertEqual(_clean_table_cell(None), "")
        self.assertEqual(_clean_table_cell(123), "123")

        # 2. Test table formatting with pipe and newline injections
        injected_analysis = {
            "resume_id": "res_bad",
            "jd_id": "jd_bad",
            "context_sufficient": True,
            "match_score": 75,
            "summary": "Candidate profile summary.",
            "strengths": ["Skill A | B"],
            "weaknesses": ["Weakness 1\nnewline"],
            "skill_gap": {
                "matched": [{"skill_name": "C++ | C#"}],
                "missing_essential": ["CI/CD | Jenkins"],
                "missing_preferred": ["Docker\nContainerization"],
                "extra": ["Linux | Bash"]
            },
            "requirement_match": [
                {"category": "Skills | Core", "match_percent": 75}
            ],
            "recommendations": [
                {"skill": "CI/CD | Jenkins", "reason": "Deployments | Automations\nRequired"}
            ]
        }

        report = generate_markdown_report(injected_analysis, "Candidate | Name", "Role | Title")
        # Ensure pipes are escaped in cells
        self.assertIn(r"Candidate \| Name", report)
        self.assertIn(r"Role \| Title", report)
        self.assertIn(r"C++ \| C#", report)
        self.assertIn(r"CI/CD \| Jenkins", report)
        self.assertIn(r"Skills \| Core", report)
        self.assertIn(r"Deployments \| Automations Required", report)
        # Ensure raw newlines inside cells were flattened
        self.assertNotIn("Docker\nContainerization", report)

        # 3. Test insufficient context handling
        insuff_analysis = {
            "resume_id": "res_short",
            "jd_id": "jd_short",
            "context_sufficient": False,
            "insufficient_reason": "Resume text contains only 12 words (minimum 30 words required).",
            "match_score": 0,
            "summary": "Analysis bypassed due to insufficient document context.",
            "strengths": [],
            "weaknesses": [],
            "skill_gap": {},
            "requirement_match": [],
            "recommendations": []
        }

        insuff_report = generate_markdown_report(insuff_analysis, "Short Resume", "Junior Dev")
        self.assertIn("**Context Sufficient:** No", insuff_report)
        self.assertIn("Context Insufficient Gate Triggered", insuff_report)
        self.assertIn("Resume text contains only 12 words", insuff_report)
        self.assertIn("Automated LLM generation was safely bypassed.", insuff_report)

    def test_matches_work_arrangement_heuristics(self):
        """Test work arrangement filtering heuristic edge cases."""
        from app import matches_work_arrangement

        # 1. None checked -> all pass
        self.assertTrue(matches_work_arrangement("Remote", False, False, False))
        self.assertTrue(matches_work_arrangement("Hybrid - London", False, False, False))
        self.assertTrue(matches_work_arrangement("Chicago, IL", False, False, False))
        self.assertTrue(matches_work_arrangement("", False, False, False))

        # 2. Remote checked
        self.assertTrue(matches_work_arrangement("Remote, USA", True, False, False))
        self.assertTrue(matches_work_arrangement("Fully Remote", True, False, False))
        self.assertFalse(matches_work_arrangement("Chicago, IL", True, False, False))
        self.assertFalse(matches_work_arrangement("On-site New York", True, False, False))

        # 3. Hybrid checked
        self.assertTrue(matches_work_arrangement("Hybrid - Boston", False, True, False))
        self.assertFalse(matches_work_arrangement("Remote, USA", False, True, False))
        self.assertFalse(matches_work_arrangement("Chicago, IL", False, True, False))

        # 4. On-site checked
        self.assertTrue(matches_work_arrangement("On-site Seattle", False, False, True))
        self.assertTrue(matches_work_arrangement("In-person Seattle", False, False, True))
        self.assertTrue(matches_work_arrangement("Office based Austin", False, False, True))
        # Physical location without remote or hybrid keyword should count as On-site
        self.assertTrue(matches_work_arrangement("Chicago, IL", False, False, True))
        self.assertTrue(matches_work_arrangement("London, UK", False, False, True))
        # Explicit remote should not match on-site
        self.assertFalse(matches_work_arrangement("Remote, Worldwide", False, False, True))
        # Hybrid should not match purely on-site
        self.assertFalse(matches_work_arrangement("Hybrid - San Francisco", False, False, True))

        # 5. Multiple checkboxes (Remote OR On-site)
        self.assertTrue(matches_work_arrangement("Remote", True, False, True))
        self.assertTrue(matches_work_arrangement("Chicago, IL", True, False, True))
        self.assertFalse(matches_work_arrangement("Hybrid - Denver", True, False, True))

    def test_composite_caching_isolation(self):
        """Test composite caching helpers isolate analyses across candidate resumes."""
        import streamlit as st
        from app import get_analysis_cache_key, get_cached_analysis, set_cached_analysis

        # Mock session state
        st.session_state["analyses"] = {}

        cand1_id = "cand_alpha"
        cand2_id = "cand_beta"
        jd_id = "jd_999"

        key1 = get_analysis_cache_key(cand1_id, jd_id)
        key2 = get_analysis_cache_key(cand2_id, jd_id)
        self.assertNotEqual(key1, key2)
        self.assertEqual(key1, "cand_alpha::jd_999")
        self.assertEqual(key2, "cand_beta::jd_999")

        # Initially both are None
        self.assertIsNone(get_cached_analysis(cand1_id, jd_id))
        self.assertIsNone(get_cached_analysis(cand2_id, jd_id))

        # Set analysis for candidate 1
        analysis_cand1 = {"resume_id": cand1_id, "jd_id": jd_id, "match_score": 92}
        set_cached_analysis(cand1_id, jd_id, analysis_cand1)

        # Candidate 1 should have analysis, Candidate 2 should NOT get Candidate 1's analysis
        self.assertEqual(get_cached_analysis(cand1_id, jd_id), analysis_cand1)
        self.assertIsNone(get_cached_analysis(cand2_id, jd_id))

        # Set distinct analysis for candidate 2
        analysis_cand2 = {"resume_id": cand2_id, "jd_id": jd_id, "match_score": 65}
        set_cached_analysis(cand2_id, jd_id, analysis_cand2)

        # Both retrieve their own distinct analyses
        self.assertEqual(get_cached_analysis(cand1_id, jd_id)["match_score"], 92)
        self.assertEqual(get_cached_analysis(cand2_id, jd_id)["match_score"], 65)

    def test_plotly_charts_empty_guards(self):
        """Test Plotly helper charts never crash on empty, None, or zero data."""
        from app import (
            make_skill_donut_chart,
            make_requirement_bar_chart,
            make_skill_comparison_bar,
            make_multi_jd_grouped_bar,
            make_skill_gap_dashboard_chart
        )

        # Donut chart
        fig_donut_empty = make_skill_donut_chart({})
        self.assertIsNotNone(fig_donut_empty)
        fig_donut_none = make_skill_donut_chart(None)
        self.assertIsNotNone(fig_donut_none)

        # Requirement bar chart
        fig_req_empty = make_requirement_bar_chart([])
        self.assertIsNotNone(fig_req_empty)
        fig_req_none = make_requirement_bar_chart(None)
        self.assertIsNotNone(fig_req_none)

        # Skill volume comparison bar
        fig_vol_empty = make_skill_comparison_bar({})
        self.assertIsNotNone(fig_vol_empty)
        fig_vol_none = make_skill_comparison_bar(None)
        self.assertIsNotNone(fig_vol_none)

        # Multi-JD comparison grouped bar
        fig_multi_empty = make_multi_jd_grouped_bar({}, {})
        self.assertIsNotNone(fig_multi_empty)

        # Dashboard chart
        fig_dash_empty = make_skill_gap_dashboard_chart({}, {})
        self.assertIsNotNone(fig_dash_empty)

    def test_ranking_tie_breaking_and_edge_cases(self):
        """Test deterministic tie-breaking by jd_id and synonym arrangement boosting."""
        jd_index.clear_store_for_testing()

        fake_vec = [0.5] * 3072
        with patch.object(jd_index, "_get_embedder") as mock_emb_fn:
            mock_emb = MagicMock()
            mock_emb.embed_chunk.return_value = fake_vec
            mock_emb_fn.return_value = mock_emb

            # Add two JDs that will have identical raw cosine similarity
            rec_b = jd_index.add_jd("Role B", "Company B", "Denver, In-person", "Text content " * 10)
            rec_a = jd_index.add_jd("Role A", "Company A", "Austin, Office", "Text content " * 10)

        with patch.object(ranking, "_get_embedder") as mock_rank_emb_fn:
            mock_rank_emb = MagicMock()
            mock_rank_emb.embed_chunk.return_value = fake_vec
            mock_rank_emb_fn.return_value = mock_rank_emb

            # 1. With On-site preference, both get +0.03 boost from synonyms ("in-person", "office")
            # Raw similarity is equal (1.0 -> 100).
            # Tie-breaker sorts by jd_id deterministically
            res = ranking.rank_jds("Resume text", "res_tie", preferred_arrangement="On-site")
            self.assertEqual(len(res.results), 2)
            # Scores should be clamped to 100
            self.assertEqual(res.results[0].similarity_score, 100)
            self.assertEqual(res.results[1].similarity_score, 100)

            # 2. Test empty string preferences are ignored
            res_empty_pref = ranking.rank_jds("Resume text", "res_tie", preferred_location="   ", preferred_arrangement="Any")
            self.assertEqual(len(res_empty_pref.results), 2)


if __name__ == "__main__":
    unittest.main()

