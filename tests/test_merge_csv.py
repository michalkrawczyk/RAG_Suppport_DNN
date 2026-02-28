"""Tests for CSV merger."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from RAG_supporters.data_prep import CSVMerger, merge_csv_files


class TestCSVMergerInit:
    """Test CSVMerger initialization."""

    def test_default_aliases(self):
        """Test CSVMerger initializes with default aliases."""
        merger = CSVMerger()

        assert "question" in merger.column_aliases, "Should have 'question' aliases"
        assert "source" in merger.column_aliases, "Should have 'source' aliases"
        assert "answer" in merger.column_aliases, "Should have 'answer' aliases"
        assert "keywords" in merger.column_aliases, "Should have 'keywords' aliases"
        assert "relevance_score" in merger.column_aliases, "Should have 'relevance_score' aliases"

    def test_custom_aliases(self):
        """Test CSVMerger accepts custom aliases."""
        custom_aliases = {"question": ["q", "query"], "source": ["s", "doc"]}

        merger = CSVMerger(column_aliases=custom_aliases)

        assert merger.column_aliases == custom_aliases, "Should use custom aliases"


class TestCSVMergerColumnFinding:
    """Test column finding with aliases."""

    def test_find_column_exact_match(self):
        """Test _find_column() finds exact match."""
        merger = CSVMerger()
        df = pd.DataFrame({"question": ["q1"], "source": ["s1"]})

        found = merger._find_column(df, "question")

        assert found == "question", "Should find exact column name match"

    def test_find_column_alias_match(self):
        """Test _find_column() finds alias match."""
        merger = CSVMerger()
        df = pd.DataFrame({"question_text": ["q1"], "source": ["s1"]})

        found = merger._find_column(df, "question")

        assert found == "question_text", "Should find column using alias"

    def test_find_column_no_match(self):
        """Test _find_column() returns None when no match."""
        merger = CSVMerger()
        df = pd.DataFrame({"other": ["value"]})

        found = merger._find_column(df, "question")

        assert found is None, "Should return None when column not found"


class TestCSVMergerNormalization:
    """Test DataFrame normalization."""

    def test_normalize_minimal_columns(self):
        """Test normalization with only required columns."""
        merger = CSVMerger()
        df = pd.DataFrame(
            {
                "question": ["What is Python?", "What is Java?"],
                "source": ["Python is a language", "Java is a language"],
            }
        )

        normalized = merger._normalize_dataframe(df, "test.csv")

        assert "question" in normalized.columns, "Should have 'question' column"
        assert "source" in normalized.columns, "Should have 'source' column"
        assert "answer" in normalized.columns, "Should have 'answer' column with empty values"
        assert "keywords" in normalized.columns, "Should have 'keywords' column with empty lists"
        assert (
            "relevance_score" in normalized.columns
        ), "Should have 'relevance_score' column with 1.0"
        assert len(normalized) == 2, "Should preserve all rows"

    def test_normalize_all_columns(self):
        """Test normalization with all columns present."""
        merger = CSVMerger()
        df = pd.DataFrame(
            {
                "question": ["What is Python?"],
                "source": ["Python is a language"],
                "answer": ["Python is a programming language"],
                "keywords": ["python, programming"],
                "relevance_score": [0.95],
            }
        )

        normalized = merger._normalize_dataframe(df, "test.csv")

        assert (
            normalized["answer"].iloc[0] == "Python is a programming language"
        ), "Should preserve answer"
        assert len(normalized["keywords"].iloc[0]) > 0, "Should have parsed keywords"
        assert normalized["relevance_score"].iloc[0] == 0.95, "Should preserve relevance score"

    def test_normalize_missing_required_column(self):
        """Test normalization raises error when required column is missing."""
        merger = CSVMerger()
        df = pd.DataFrame({"question": ["What is Python?"]})

        with pytest.raises(ValueError, match="Required column 'source' not found"):
            merger._normalize_dataframe(df, "test.csv")

    def test_normalize_with_aliases(self):
        """Test normalization works with column aliases."""
        merger = CSVMerger()
        df = pd.DataFrame(
            {
                "question_text": ["What is Python?"],
                "source_text": ["Python is a language"],
                "score": [0.8],
            }
        )

        normalized = merger._normalize_dataframe(df, "test.csv")

        assert (
            normalized["question"].iloc[0] == "What is Python?"
        ), "Should map 'question_text' to 'question'"
        assert (
            normalized["source"].iloc[0] == "Python is a language"
        ), "Should map 'source_text' to 'source'"
        assert (
            normalized["relevance_score"].iloc[0] == 0.8
        ), "Should map 'score' to 'relevance_score'"

    def test_normalize_clips_scores(self):
        """Test normalization clips relevance scores to [0, 1]."""
        merger = CSVMerger()
        df = pd.DataFrame(
            {
                "question": ["q1", "q2", "q3"],
                "source": ["s1", "s2", "s3"],
                "score": [1.5, -0.5, 0.5],  # Out of range
            }
        )

        normalized = merger._normalize_dataframe(df, "test.csv")

        assert normalized["relevance_score"].iloc[0] == 1.0, "Should clip score > 1 to 1.0"
        assert normalized["relevance_score"].iloc[1] == 0.0, "Should clip score < 0 to 0.0"
        assert normalized["relevance_score"].iloc[2] == 0.5, "Should preserve score in range"


class TestCSVMergerKeywordParsing:
    """Test keyword parsing from various formats."""

    def test_parse_keywords_comma_separated(self):
        """Test parsing comma-separated keywords."""
        merger = CSVMerger()

        result = merger._parse_keywords("python, programming, language")

        assert result == [
            "python",
            "programming",
            "language",
        ], "Should parse comma-separated keywords"

    def test_parse_keywords_json_list(self):
        """Test parsing JSON list of keywords."""
        merger = CSVMerger()

        result = merger._parse_keywords('["python", "programming", "language"]')

        assert result == ["python", "programming", "language"], "Should parse JSON list keywords"

    def test_parse_keywords_single_string(self):
        """Test parsing single keyword string."""
        merger = CSVMerger()

        result = merger._parse_keywords("python")

        assert result == ["python"], "Should parse single keyword"

    def test_parse_keywords_empty(self):
        """Test parsing empty keywords."""
        merger = CSVMerger()

        result = merger._parse_keywords("")

        assert result == [], "Should return empty list for empty string"

    def test_parse_keywords_nan(self):
        """Test parsing NaN keywords."""
        merger = CSVMerger()

        result = merger._parse_keywords(pd.NA)

        assert result == [], "Should return empty list for NaN"

    def test_parse_keywords_list(self):
        """Test parsing list of keywords."""
        merger = CSVMerger()

        result = merger._parse_keywords(["python", "programming"])

        assert result == ["python", "programming"], "Should handle list input"


class TestCSVMergerDeduplication:
    """Test duplicate merging logic."""

    def test_merge_duplicates_no_duplicates(self):
        """Test merge_duplicates() preserves unique pairs."""
        merger = CSVMerger()
        df = pd.DataFrame(
            {
                "question": ["q1", "q2"],
                "source": ["s1", "s2"],
                "answer": ["a1", "a2"],
                "keywords": [["k1"], ["k2"]],
                "relevance_score": [0.9, 0.8],
            }
        )

        merged = merger._merge_duplicates(df)

        assert len(merged) == 2, "Should preserve all unique pairs"

    def test_merge_duplicates_many_to_many_preserved(self):
        """Test merge_duplicates() preserves many-to-many relationships."""
        merger = CSVMerger()
        df = pd.DataFrame(
            {
                "question": ["q1", "q1", "q2", "q2"],
                "source": ["s1", "s2", "s1", "s2"],
                "answer": ["a1", "a2", "a3", "a4"],
                "keywords": [["k1"], ["k2"], ["k3"], ["k4"]],
                "relevance_score": [0.9, 0.8, 0.85, 0.95],
            }
        )

        merged = merger._merge_duplicates(df)

        assert len(merged) == 4, "Should preserve all unique pairs in many-to-many relationship"
        # q1 should have 2 sources
        q1_pairs = merged[merged["question"] == "q1"]
        assert len(q1_pairs) == 2, "Question q1 should have 2 different sources"
        assert set(q1_pairs["source"]) == {
            "s1",
            "s2",
        }, "Question q1 should be paired with s1 and s2"

        # s1 should be paired with 2 questions
        s1_pairs = merged[merged["source"] == "s1"]
        assert len(s1_pairs) == 2, "Source s1 should be paired with 2 different questions"
        assert set(s1_pairs["question"]) == {
            "q1",
            "q2",
        }, "Source s1 should be paired with q1 and q2"

    def test_merge_duplicates_exact_duplicates(self):
        """Test merge_duplicates() removes exact duplicates."""
        merger = CSVMerger()
        df = pd.DataFrame(
            {
                "question": ["q1", "q1"],
                "source": ["s1", "s1"],
                "answer": ["a1", "a1"],
                "keywords": [["k1"], ["k1"]],
                "relevance_score": [0.9, 0.9],
            }
        )

        merged = merger._merge_duplicates(df)

        assert len(merged) == 1, "Should merge exact duplicate pairs to single row"

    def test_merge_duplicates_max_score(self):
        """Test merge_duplicates() keeps max relevance score."""
        merger = CSVMerger()
        df = pd.DataFrame(
            {
                "question": ["q1", "q1"],
                "source": ["s1", "s1"],
                "answer": ["a1", "a2"],
                "keywords": [["k1"], ["k2"]],
                "relevance_score": [0.7, 0.9],
            }
        )

        merged = merger._merge_duplicates(df)

        assert merged["relevance_score"].iloc[0] == 0.9, "Should keep maximum relevance score"

    def test_merge_duplicates_union_keywords(self):
        """Test merge_duplicates() unions keywords."""
        merger = CSVMerger()
        df = pd.DataFrame(
            {
                "question": ["q1", "q1"],
                "source": ["s1", "s1"],
                "answer": ["a1", "a2"],
                "keywords": [["k1", "k2"], ["k2", "k3"]],
                "relevance_score": [0.9, 0.8],
            }
        )

        merged = merger._merge_duplicates(df)

        keywords = set(merged["keywords"].iloc[0])
        assert keywords == {"k1", "k2", "k3"}, "Should union all unique keywords"

    def test_merge_duplicates_longest_answer(self):
        """Test merge_duplicates() keeps longest answer."""
        merger = CSVMerger()
        df = pd.DataFrame(
            {
                "question": ["q1", "q1"],
                "source": ["s1", "s1"],
                "answer": ["short", "long answer text"],
                "keywords": [["k1"], ["k2"]],
                "relevance_score": [0.9, 0.8],
            }
        )

        merged = merger._merge_duplicates(df)

        assert merged["answer"].iloc[0] == "long answer text", "Should keep longest answer"


class TestCSVMergerIDAssignment:
    """Test ID assignment logic."""

    def test_assign_ids_unique_questions(self):
        """Test _assign_ids() creates unique question IDs."""
        merger = CSVMerger()
        df = pd.DataFrame(
            {
                "question": ["q1", "q2", "q1"],
                "source": ["s1", "s2", "s3"],
                "answer": ["", "", ""],
                "keywords": [[], [], []],
                "relevance_score": [1.0, 1.0, 1.0],
            }
        )

        result = merger._assign_ids(df)

        assert "question_id" in result.columns, "Should have 'question_id' column"
        assert result["question_id"].nunique() == 2, "Should have 2 unique question IDs"
        assert (
            result.loc[result["question"] == "q1", "question_id"].nunique() == 1
        ), "Same question should have same ID"

    def test_assign_ids_unique_sources(self):
        """Test _assign_ids() creates unique source IDs."""
        merger = CSVMerger()
        df = pd.DataFrame(
            {
                "question": ["q1", "q2", "q3"],
                "source": ["s1", "s2", "s1"],
                "answer": ["", "", ""],
                "keywords": [[], [], []],
                "relevance_score": [1.0, 1.0, 1.0],
            }
        )

        result = merger._assign_ids(df)

        assert "source_id" in result.columns, "Should have 'source_id' column"
        assert result["source_id"].nunique() == 2, "Should have 2 unique source IDs"
        assert (
            result.loc[result["source"] == "s1", "source_id"].nunique() == 1
        ), "Same source should have same ID"

    def test_assign_ids_sequential_pairs(self):
        """Test _assign_ids() creates sequential pair IDs."""
        merger = CSVMerger()
        df = pd.DataFrame(
            {
                "question": ["q1", "q2", "q3"],
                "source": ["s1", "s2", "s3"],
                "answer": ["", "", ""],
                "keywords": [[], [], []],
                "relevance_score": [1.0, 1.0, 1.0],
            }
        )

        result = merger._assign_ids(df)

        assert "pair_id" in result.columns, "Should have 'pair_id' column"
        assert list(result["pair_id"]) == [0, 1, 2], "Pair IDs should be sequential starting from 0"


class TestCSVMergerIntegration:
    """Test full merge workflow."""

    def test_merge_single_file(self):
        """Test merging a single CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            df = pd.DataFrame(
                {
                    "question": ["What is Python?", "What is Java?"],
                    "source": ["Python is a language", "Java is a language"],
                    "score": [0.9, 0.8],
                }
            )
            df.to_csv(csv_path, index=False)

            merger = CSVMerger()
            result = merger.merge_csv_files([csv_path])

            assert len(result) == 2, "Should have 2 pairs"
            assert "question_id" in result.columns, "Should have question_id column"
            assert "source_id" in result.columns, "Should have source_id column"
            assert "pair_id" in result.columns, "Should have pair_id column"

    def test_merge_multiple_files(self):
        """Test merging multiple CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # File 1
            csv1 = Path(tmpdir) / "data1.csv"
            df1 = pd.DataFrame(
                {"question": ["What is Python?"], "source": ["Python is a language"]}
            )
            df1.to_csv(csv1, index=False)

            # File 2
            csv2 = Path(tmpdir) / "data2.csv"
            df2 = pd.DataFrame({"question": ["What is Java?"], "source": ["Java is a language"]})
            df2.to_csv(csv2, index=False)

            merger = CSVMerger()
            result = merger.merge_csv_files([csv1, csv2])

            assert len(result) == 2, "Should combine rows from both files"

    def test_merge_many_to_many_workflow(self):
        """Test complete merge workflow preserves many-to-many relationships."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            df = pd.DataFrame(
                {
                    "question": [
                        "What is Python?",
                        "What is Python?",  # Same question, different source
                        "What is Java?",
                        "What is Java?",  # Same question, different source
                    ],
                    "source": [
                        "Python is a language",
                        "Python is used for AI",
                        "Python is a language",  # Same source, different question
                        "Java is a language",
                    ],
                    "score": [0.9, 0.85, 0.8, 0.95],
                }
            )
            df.to_csv(csv_path, index=False)

            merger = CSVMerger()
            result = merger.merge_csv_files([csv_path])

            # Should have 4 unique pairs (many-to-many preserved)
            assert (
                len(result) == 4
            ), "Should preserve all 4 unique pairs in many-to-many relationship"

            # Check question IDs
            assert result["question_id"].nunique() == 2, "Should have 2 unique questions"

            # Check source IDs
            assert result["source_id"].nunique() == 3, "Should have 3 unique sources"

            # Verify specific relationships
            q_python = result[result["question"] == "What is Python?"]
            assert len(q_python) == 2, "Python question should have 2 sources"

            s_python_lang = result[result["source"] == "Python is a language"]
            assert len(s_python_lang) == 2, "Python language source should answer 2 questions"

    def test_merge_with_exact_duplicates_in_many_to_many(self):
        """Test merge removes exact duplicates while preserving many-to-many."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            df = pd.DataFrame(
                {
                    "question": [
                        "What is Python?",
                        "What is Python?",  # Duplicate pair
                        "What is Python?",  # Different source (many-to-many)
                        "What is Java?",
                    ],
                    "source": [
                        "Python is a language",
                        "Python is a language",  # Duplicate pair
                        "Python is used for AI",  # Different source
                        "Python is a language",  # Different question
                    ],
                    "score": [0.9, 0.85, 0.8, 0.75],
                }
            )
            df.to_csv(csv_path, index=False)

            merger = CSVMerger()
            result = merger.merge_csv_files([csv_path])

            # Should have 3 pairs after removing 1 duplicate
            assert len(result) == 3, "Should have 3 pairs after merging 1 duplicate"

            # Verify the duplicate was merged with max score
            python_lang_pair = result[
                (result["question"] == "What is Python?")
                & (result["source"] == "Python is a language")
            ]
            assert len(python_lang_pair) == 1, "Duplicate pair should be merged to single row"
            assert (
                python_lang_pair["relevance_score"].iloc[0] == 0.9
            ), "Should keep max score from duplicates"

    def test_merge_with_output_path(self):
        """Test merging saves to output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "input.csv"
            df = pd.DataFrame({"question": ["What is Python?"], "source": ["Python is a language"]})
            df.to_csv(csv_path, index=False)

            output_path = Path(tmpdir) / "output.csv"

            merger = CSVMerger()
            merger.merge_csv_files([csv_path], output_path=output_path)

            assert output_path.exists(), "Output file should be created"
            loaded = pd.read_csv(output_path)
            assert len(loaded) == 1, "Output file should contain merged data"

    def test_merge_removes_empty_rows(self):
        """Test merging removes rows with empty question or source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            df = pd.DataFrame(
                {
                    "question": ["What is Python?", "", "What is Java?"],
                    "source": ["Python is a language", "Some source", ""],
                }
            )
            df.to_csv(csv_path, index=False)

            merger = CSVMerger()
            result = merger.merge_csv_files([csv_path])

            assert len(result) == 1, "Should remove rows with empty question or source"
            assert result["question"].iloc[0] == "What is Python?", "Should keep valid row"


class TestCSVMergerConvenienceFunction:
    """Test convenience function."""

    def test_merge_csv_files_function(self):
        """Test merge_csv_files() convenience function works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            df = pd.DataFrame({"question": ["What is Python?"], "source": ["Python is a language"]})
            df.to_csv(csv_path, index=False)

            result = merge_csv_files([csv_path])

            assert len(result) == 1, "Convenience function should work"
            assert "pair_id" in result.columns, "Should have pair_id column"


class TestCSVMergerInspectionMetadata:
    """Test inspection metadata creation."""

    def test_create_inspection_metadata(self):
        """Test create_inspection_metadata() generates correct structure."""
        merger = CSVMerger()
        df = pd.DataFrame(
            {
                "pair_id": [0, 1],
                "question_id": [0, 1],
                "source_id": [0, 1],
                "question": ["What is Python?", "What is Java?"],
                "source": ["Python is a language", "Java is a language"],
                "answer": ["", ""],
                "keywords": [["python"], []],
                "relevance_score": [0.9, 0.8],
            }
        )

        metadata = merger.create_inspection_metadata(df, "clusters.json")

        assert "metadata" in metadata, "Should have 'metadata' key"
        assert "questions" in metadata, "Should have 'questions' key"
        assert "sources" in metadata, "Should have 'sources' key"
        assert metadata["metadata"]["n_pairs"] == 2, "Should have correct pair count"
        assert metadata["metadata"]["n_questions"] == 2, "Should have correct question count"
        assert (
            metadata["metadata"]["clustering_source"] == "clusters.json"
        ), "Should reference clustering source"


class TestCSVMergerEdgeCases:
    """Test edge cases and error handling."""

    def test_merge_empty_file_list(self):
        """Test merge raises error for empty file list."""
        merger = CSVMerger()

        with pytest.raises(ValueError, match="No CSV files provided"):
            merger.merge_csv_files([])

    def test_merge_nonexistent_file(self):
        """Test merge raises error for non-existent file."""
        merger = CSVMerger()

        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            merger.merge_csv_files(["nonexistent.csv"])


class TestCSVMergerSuggestionParsing:
    """Test _parse_keywords with extract_suggestions list-of-dicts format."""

    def test_parse_suggestions_list_of_dicts(self):
        """Python List[dict] with 'term' key extracts term strings."""
        merger = CSVMerger()
        value = [
            {"term": "Medulloblastoma", "type": "domain", "confidence": 0.99, "reason": "x"},
            {"term": "Copy number variation", "type": "keyword", "confidence": 0.95, "reason": "y"},
        ]
        result = merger._parse_keywords(value)
        assert result == ["medulloblastoma", "copy number variation"]

    def test_parse_suggestions_json_string(self):
        """JSON-serialised '[{\"term\": ...}]' string extracts terms."""
        import json

        merger = CSVMerger()
        value = json.dumps(
            [
                {"term": "Machine Learning", "type": "domain", "confidence": 0.9, "reason": ""},
                {"term": "Neural Network", "type": "keyword", "confidence": 0.8, "reason": ""},
            ]
        )
        result = merger._parse_keywords(value)
        assert "machine learning" in result
        assert "neural network" in result

    def test_parse_suggestions_confidence_filter(self):
        """suggestion_min_confidence=0.95 removes low-confidence terms."""
        merger = CSVMerger(suggestion_min_confidence=0.95)
        value = [
            {"term": "High Confidence", "type": "keyword", "confidence": 0.99},
            {"term": "Low Confidence", "type": "keyword", "confidence": 0.5},
        ]
        result = merger._parse_keywords(value)
        assert "high confidence" in result
        assert "low confidence" not in result

    def test_parse_suggestions_type_filter(self):
        """suggestion_types=['keyword'] keeps only matching type entries."""
        merger = CSVMerger(suggestion_types=["keyword"])
        value = [
            {"term": "Domain Term", "type": "domain", "confidence": 0.9},
            {"term": "Keyword Term", "type": "keyword", "confidence": 0.9},
        ]
        result = merger._parse_keywords(value)
        assert "keyword term" in result
        assert "domain term" not in result

    def test_parse_suggestions_combined_filter(self):
        """Both confidence and type filtering applied together."""
        merger = CSVMerger(suggestion_min_confidence=0.8, suggestion_types=["keyword"])
        value = [
            {"term": "A", "type": "keyword", "confidence": 0.9},  # passes both
            {"term": "B", "type": "keyword", "confidence": 0.5},  # fails confidence
            {"term": "C", "type": "domain", "confidence": 0.9},  # fails type
            {"term": "D", "type": "domain", "confidence": 0.5},  # fails both
        ]
        result = merger._parse_keywords(value)
        assert result == ["a"]

    def test_parse_suggestions_empty_after_filter(self):
        """All items filtered → returns []."""
        merger = CSVMerger(suggestion_min_confidence=0.99)
        value = [
            {"term": "Low", "type": "keyword", "confidence": 0.1},
        ]
        result = merger._parse_keywords(value)
        assert result == []

    def test_extract_suggestions_column_alias(self, tmp_path):
        """Full merge_csv_files call with extract_suggestions column → keywords populated."""
        import json

        df = pd.DataFrame(
            {
                "question": ["What is ML?"],
                "source": ["ML is a subset of AI."],
                "extract_suggestions": [
                    json.dumps(
                        [
                            {
                                "term": "Machine Learning",
                                "type": "keyword",
                                "confidence": 0.9,
                                "reason": "",
                            }
                        ]
                    )
                ],
            }
        )
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        from RAG_supporters.data_prep import merge_csv_files as _merge

        result = _merge([str(csv_path)])

        assert len(result) == 1
        kws = result.iloc[0]["keywords"]
        assert isinstance(kws, list)
        assert "machine learning" in kws

    def test_backward_compat_plain_list(self):
        """Existing plain ['kw1', 'kw2'] list still parses correctly."""
        merger = CSVMerger()
        result = merger._parse_keywords(["kw1", "kw2"])
        assert result == ["kw1", "kw2"]

    def test_backward_compat_csv_string(self):
        """Existing 'kw1, kw2' string still parses correctly."""
        merger = CSVMerger()
        result = merger._parse_keywords("kw1, kw2")
        assert result == ["kw1", "kw2"]


class TestCSVMergerTopicRelevanceParsing:
    """Test _parse_keywords with topic_relevance_prob_topic_scores list-of-dicts format."""

    def test_parse_topic_scores_list_of_dicts(self):
        """Python List[dict] with 'topic_descriptor' key → extracts descriptors."""
        merger = CSVMerger(topic_min_probability=0.0)
        value = [
            {"topic_descriptor": "cancer targeting", "probability": 0.75, "reason": None},
            {"topic_descriptor": "cancer progression", "probability": 0.85, "reason": None},
        ]
        result = merger._parse_keywords(value)
        assert "cancer targeting" in result
        assert "cancer progression" in result

    def test_parse_topic_scores_json_string(self):
        """JSON-serialised topic scores string → extracts descriptors."""
        import json

        merger = CSVMerger(topic_min_probability=0.0)
        value = json.dumps(
            [
                {"topic_descriptor": "deep learning", "probability": 0.9, "reason": None},
                {"topic_descriptor": "computer vision", "probability": 0.7, "reason": None},
            ]
        )
        result = merger._parse_keywords(value)
        assert "deep learning" in result
        assert "computer vision" in result

    def test_parse_topic_scores_probability_filter(self):
        """topic_min_probability=0.7 removes low-probability topics."""
        merger = CSVMerger(topic_min_probability=0.7)
        value = [
            {"topic_descriptor": "High Prob", "probability": 0.9},
            {"topic_descriptor": "Low Prob", "probability": 0.4},
        ]
        result = merger._parse_keywords(value)
        assert "high prob" in result
        assert "low prob" not in result

    def test_parse_topic_scores_empty_after_filter(self):
        """All topics below threshold → returns []."""
        merger = CSVMerger(topic_min_probability=0.99)
        value = [
            {"topic_descriptor": "anything", "probability": 0.3},
        ]
        result = merger._parse_keywords(value)
        assert result == []

    def test_parse_topic_scores_partial_coverage(self):
        """Incomplete column still returns non-empty list when above threshold."""
        merger = CSVMerger(topic_min_probability=0.5)
        value = [
            {"topic_descriptor": "main topic", "probability": 0.8},
        ]
        result = merger._parse_keywords(value)
        assert len(result) >= 1
        assert "main topic" in result

    def test_topic_relevance_column_alias(self, tmp_path):
        """Full merge with topic_relevance_prob_topic_scores column → keywords populated."""
        import json

        df = pd.DataFrame(
            {
                "question": ["What causes cancer?"],
                "source": ["Cancer is caused by mutations."],
                "topic_relevance_prob_topic_scores": [
                    json.dumps(
                        [{"topic_descriptor": "oncology", "probability": 0.85, "reason": None}]
                    )
                ],
            }
        )
        csv_path = tmp_path / "input.csv"
        df.to_csv(csv_path, index=False)
        from RAG_supporters.data_prep import merge_csv_files as _merge

        result = _merge([str(csv_path)], topic_min_probability=0.5)

        assert len(result) == 1
        kws = result.iloc[0]["keywords"]
        assert isinstance(kws, list)
        assert "oncology" in kws

    def test_unknown_dict_schema(self):
        """Dict with neither 'term' nor 'topic_descriptor' → returns []."""
        merger = CSVMerger()
        value = [{"foo": "bar", "baz": 1.0}]
        result = merger._parse_keywords(value)
        assert result == []
