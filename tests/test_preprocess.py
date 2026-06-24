"""
Testes unitários para src.pipeline.preprocess.
"""

from src.pipeline.preprocess import clean_text, tokenize


class TestCleanText:
    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_input(self):
        assert clean_text(None) == ""
        assert clean_text(123) == ""
        assert clean_text([]) == ""

    def test_lowercase_conversion(self):
        result = clean_text("DATA SCIENCE")
        assert result == result.lower()

    def test_remove_punctuation(self):
        result = clean_text("python, sql! machine; learning:")
        assert "," not in result
        assert "!" not in result
        assert ";" not in result
        assert ":" not in result

    def test_remove_stopwords(self):
        result = clean_text("the and of a in data science")
        assert "the" not in result
        assert "and" not in result
        assert "of" not in result

    def test_remove_short_tokens(self):
        result = clean_text("a an to be data science")
        assert "data" in result
        assert "science" in result

    def test_lemmatization(self):
        result = clean_text("running studies analyses")
        assert "running" in result
        assert "study" in result
        assert "analysis" in result or "analyses" in result

    def test_normal_text(self):
        text = "Data Scientist with 5 years of experience in Python and SQL!"
        result = clean_text(text)
        assert "data" in result
        assert "scientist" in result
        assert "python" in result
        assert "sql" in result
        assert "5" in result or "year" in result
        assert "experience" in result


class TestTokenize:
    def test_empty_string(self):
        assert tokenize("") == []

    def test_none_input(self):
        assert tokenize(None) == []

    def test_returns_list(self):
        tokens = tokenize("data science python")
        assert isinstance(tokens, list)
        assert len(tokens) == 3

    def test_no_stopwords_in_output(self):
        tokens = tokenize("the and data of science in python")
        assert "the" not in tokens
        assert "and" not in tokens
