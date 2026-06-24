"""
Testes unitários para src.skills.skills_analyzer.
"""

from src.skills.skills_analyzer import (
    _generate_plan,
    analyze_gap,
    extract_skills_from_text,
)


class TestExtractSkillsFromText:
    def test_find_skills(self):
        text = "I know Python, SQL and Machine Learning"
        vocab = {"python", "sql", "machine learning", "java"}
        found = extract_skills_from_text(text, vocab)
        assert "python" in found
        assert "sql" in found
        assert "machine learning" in found
        assert "java" not in found

    def test_word_boundary_no_false_positive(self):
        text = "I know matplotlib"
        vocab = {"mat", "plot", "matplotlib"}
        found = extract_skills_from_text(text, vocab)
        assert "matplotlib" in found
        assert "mat" not in found  # word boundary

    def test_empty_text(self):
        assert extract_skills_from_text("", {"python"}) == set()

    def test_case_insensitive(self):
        text = "PYTHON and Sql"
        found = extract_skills_from_text(text, {"python", "sql"})
        assert len(found) == 2


class TestAnalyzeGap:
    def test_compatible_and_missing(self, sample_skills_map):
        resume = "I know Python and SQL"
        gap = analyze_gap(resume, "Data Scientist", sample_skills_map)
        assert "python" in gap["compatible"]
        assert "sql" in gap["compatible"]
        assert len(gap["missing"]) > 0
        assert len(gap["development_plan"]) > 0

    def test_no_skills_map_file(self, tmp_path):
        fake_path = tmp_path / "nonexistent.json"
        gap = analyze_gap("Python", "Data Scientist", fake_path)
        assert gap["compatible"] == []
        assert gap["missing"] == []
        assert gap["development_plan"] == []

    def test_unknown_title(self, sample_skills_map):
        gap = analyze_gap("Python", "Unknown Title XYZ", sample_skills_map)
        assert gap["compatible"] == []
        assert gap["missing"] == []
        assert gap["development_plan"] == []

    def test_all_skills_present(self, sample_skills_map):
        resume = "python sql machine learning statistics"
        gap = analyze_gap(resume, "Data Scientist", sample_skills_map)
        assert len(gap["missing"]) == 0
        assert len(gap["compatible"]) > 0


class TestGeneratePlan:
    def test_returns_list_of_dicts(self):
        plan = _generate_plan({"python", "docker"})
        assert len(plan) == 2
        for item in plan:
            assert "skill" in item
            assert "curso" in item
            assert "tempo" in item

    def test_known_skill(self):
        plan = _generate_plan({"python"})
        assert plan[0]["curso"] == "Python for Everybody (Coursera)"

    def test_unknown_skill(self):
        plan = _generate_plan({"extremely_obscure_tech_xyz"})
        assert "curso" in plan[0]
        assert "Coursera/Udemy" in plan[0]["curso"]

    def test_empty_set(self):
        assert _generate_plan(set()) == []
