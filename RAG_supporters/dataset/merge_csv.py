"""
CSV Merger for JASPER Steering Dataset Builder.

This module handles merging multiple CSV files with question-source pairs,
performing column normalization, deduplication, and ID assignment.

Merge Rules:
- Questions/sources are deduplicated by exact text match
- For duplicates: max relevance score, union of keywords, longest answer
- Each unique pair gets a sequential ID
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

LOGGER = logging.getLogger(__name__)


class CSVMerger:
    """Merge multiple CSV files into a unified dataset.
    
    Handles column aliasing, deduplication, and ID assignment for
    question-source pairs. Preserves many-to-many relationships:
    - One question can have multiple sources (separate pairs)
    - One source can answer multiple questions (separate pairs)
    - Only exact duplicate pairs (same question + same source) are merged
    
    Parameters
    ----------
    column_aliases : Dict[str, List[str]], optional
        Mapping from standard column names to alternative names.
        Default aliases:
        - question: ["question", "question_text", "query"]
        - source: ["source", "source_text", "context", "passage"]
        - answer: ["answer", "answer_text", "response"]
        - keywords: ["keywords", "keyword", "topics", "tags"]
        - relevance_score: ["relevance_score", "score", "relevance", "label"]
    
    Examples
    --------
    >>> merger = CSVMerger()
    >>> merged_df = merger.merge_csv_files(
    ...     csv_paths=["data1.csv", "data2.csv"],
    ...     output_path="merged.csv"
    ... )
    >>> print(f"Merged {len(merged_df)} pairs")
    >>> print(f"Questions: {merged_df['question_id'].nunique()}")
    >>> print(f"Sources: {merged_df['source_id'].nunique()}")
    """
    
    DEFAULT_ALIASES = {
        "question": ["question", "question_text", "query"],
        "source": ["source", "source_text", "context", "passage"],
        "answer": ["answer", "answer_text", "response"],
        "keywords": ["keywords", "keyword", "topics", "tags"],
        "relevance_score": ["relevance_score", "score", "relevance", "label"]
    }
    
    def __init__(
        self,
        column_aliases: Optional[Dict[str, List[str]]] = None
    ):
        """Initialize CSV merger."""
        self.column_aliases = column_aliases or self.DEFAULT_ALIASES
    
    def _find_column(self, df: pd.DataFrame, standard_name: str) -> Optional[str]:
        """Find column in DataFrame using aliases.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to search
        standard_name : str
            Standard column name (e.g., "question")
        
        Returns
        -------
        str or None
            Actual column name if found, None otherwise
        """
        aliases = self.column_aliases.get(standard_name, [])
        
        for alias in aliases:
            if alias in df.columns:
                return alias
        
        return None
    
    def _normalize_dataframe(self, df: pd.DataFrame, source_file: str) -> pd.DataFrame:
        """Normalize DataFrame to standard column names.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        source_file : str
            Original file name (for logging)
        
        Returns
        -------
        pd.DataFrame
            Normalized DataFrame with standard column names
        
        Raises
        ------
        ValueError
            If required columns (question, source) are missing
        """
        # Find required columns
        question_col = self._find_column(df, "question")
        source_col = self._find_column(df, "source")
        
        if question_col is None:
            raise ValueError(
                f"Required column 'question' not found in {source_file}. "
                f"Expected one of: {self.column_aliases['question']}"
            )
        
        if source_col is None:
            raise ValueError(
                f"Required column 'source' not found in {source_file}. "
                f"Expected one of: {self.column_aliases['source']}"
            )
        
        # Create normalized DataFrame
        normalized = pd.DataFrame({
            "question": df[question_col].astype(str),
            "source": df[source_col].astype(str)
        })
        
        # Add optional columns
        answer_col = self._find_column(df, "answer")
        if answer_col is not None:
            normalized["answer"] = df[answer_col].astype(str)
        else:
            normalized["answer"] = ""
        
        keywords_col = self._find_column(df, "keywords")
        if keywords_col is not None:
            # Parse keywords (handle JSON list or comma-separated string)
            normalized["keywords"] = df[keywords_col].apply(self._parse_keywords)
        else:
            normalized["keywords"] = [[] for _ in range(len(df))]
        
        score_col = self._find_column(df, "relevance_score")
        if score_col is not None:
            normalized["relevance_score"] = pd.to_numeric(
                df[score_col], errors="coerce"
            ).fillna(1.0)
        else:
            normalized["relevance_score"] = 1.0
        
        # Clip scores to [0, 1]
        normalized["relevance_score"] = normalized["relevance_score"].clip(0.0, 1.0)
        
        return normalized
    
    def _parse_keywords(self, value) -> List[str]:
        """Parse keywords from various formats.
        
        Handles:
        - JSON list: '["keyword1", "keyword2"]'
        - Comma-separated: "keyword1, keyword2"
        - Single string: "keyword1"
        - NaN/empty: []
        
        Parameters
        ----------
        value : any
            Keyword value from DataFrame
        
        Returns
        -------
        List[str]
            Parsed keywords
        """
        # Handle list/tuple type first (before isna check)
        if isinstance(value, (list, tuple)):
            return [str(k).strip() for k in value if k]
        
        # Check for None/NaN/empty string
        if pd.isna(value) or value == "":
            return []
        
        # Try to parse as JSON list
        if isinstance(value, str) and value.startswith("["):
            try:
                import json
                parsed = json.loads(value)
                return [str(k).strip() for k in parsed if k]
            except json.JSONDecodeError:
                pass
        
        # Parse as comma-separated
        if isinstance(value, str):
            return [k.strip() for k in value.split(",") if k.strip()]
        
        # Single value
        return [str(value).strip()]
    
    def _merge_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge duplicate question-source pairs.
        
        NOTE: Many-to-many relationships are preserved:
        - Same question + different sources = multiple pairs (NOT merged)
        - Different questions + same source = multiple pairs (NOT merged)
        - Same question + same source = duplicate pair (merged)
        
        Merge rules for duplicate pairs:
        - Max relevance score
        - Union of keywords (deduplicated)
        - Longest answer
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with potential duplicates
        
        Returns
        -------
        pd.DataFrame
            DataFrame with duplicate pairs merged, many-to-many preserved
        """
        # Group by question-source pair
        grouped = df.groupby(["question", "source"])
        
        merged_rows = []
        
        for (question, source), group in grouped:
            if len(group) == 1:
                # No duplicates
                merged_rows.append(group.iloc[0].to_dict())
            else:
                # Merge duplicate pairs (same question AND same source)
                LOGGER.debug(
                    f"Merging {len(group)} duplicate pairs for "
                    f"question='{question[:50]}...' + source='{source[:50]}...'"
                )
                
                # Max relevance score
                max_score = group["relevance_score"].max()
                
                # Union of keywords
                all_keywords: Set[str] = set()
                for keywords in group["keywords"]:
                    all_keywords.update(keywords)
                
                # Longest answer
                answers = group["answer"].tolist()
                longest_answer = max(answers, key=len)
                
                merged_rows.append({
                    "question": question,
                    "source": source,
                    "answer": longest_answer,
                    "keywords": sorted(list(all_keywords)),
                    "relevance_score": max_score
                })
        
        return pd.DataFrame(merged_rows)
    
    def _assign_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign unique IDs to questions, sources, and pairs.
        
        Parameters
        ----------
        df : pd.DataFrame
            Merged DataFrame
        
        Returns
        -------
        pd.DataFrame
            DataFrame with ID columns added:
            - question_id: Unique question ID
            - source_id: Unique source ID
            - pair_id: Unique pair ID (sequential)
        """
        # Assign question IDs
        unique_questions = df["question"].unique()
        question_to_id = {q: i for i, q in enumerate(unique_questions)}
        df["question_id"] = df["question"].map(question_to_id)
        
        # Assign source IDs
        unique_sources = df["source"].unique()
        source_to_id = {s: i for i, s in enumerate(unique_sources)}
        df["source_id"] = df["source"].map(source_to_id)
        
        # Assign pair IDs (sequential)
        df["pair_id"] = range(len(df))
        
        return df
    
    def merge_csv_files(
        self,
        csv_paths: List[Union[str, Path]],
        output_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """Merge multiple CSV files into a unified dataset.
        
        This method preserves many-to-many relationships:
        - A question can appear with multiple sources (1 question → N sources)
        - A source can appear with multiple questions (N questions → 1 source)
        - Only exact duplicate pairs (same question + same source) are merged
        
        Parameters
        ----------
        csv_paths : List[str or Path]
            Paths to CSV files to merge
        output_path : str or Path, optional
            If provided, save merged DataFrame to this path
        
        Returns
        -------
        pd.DataFrame
            Merged DataFrame with columns:
            - pair_id: Unique pair identifier
            - question_id: Unique question identifier
            - source_id: Unique source identifier
            - question: Question text
            - source: Source text
            - answer: Answer text (optional, may be empty)
            - keywords: List of keywords
            - relevance_score: Relevance score in [0, 1]
        
        Examples
        --------
        >>> merger = CSVMerger()
        >>> df = merger.merge_csv_files(
        ...     csv_paths=["data1.csv", "data2.csv"],
        ...     output_path="merged.csv"
        ... )
        >>> print(f"Total pairs: {len(df)}")
        >>> print(f"Unique questions: {df['question_id'].nunique()}")
        >>> print(f"Unique sources: {df['source_id'].nunique()}")
        >>> # Example: question with multiple sources
        >>> q1_pairs = df[df['question_id'] == 0]
        >>> print(f"Question 0 has {len(q1_pairs)} sources")
        """
        LOGGER.info(f"Merging {len(csv_paths)} CSV files")
        
        if not csv_paths:
            raise ValueError("No CSV files provided")
        
        # Load and normalize each CSV
        normalized_dfs = []
        
        for csv_path in csv_paths:
            csv_path = Path(csv_path)
            
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            LOGGER.info(f"Loading {csv_path}")
            df = pd.read_csv(csv_path)
            
            LOGGER.info(f"  Loaded {len(df)} rows")
            
            normalized = self._normalize_dataframe(df, str(csv_path))
            normalized_dfs.append(normalized)
        
        # Concatenate all DataFrames
        LOGGER.info("Concatenating DataFrames")
        combined = pd.concat(normalized_dfs, ignore_index=True)
        LOGGER.info(f"Combined: {len(combined)} rows")
        
        # Remove rows with empty question or source
        before_filter = len(combined)
        combined = combined[
            (combined["question"].str.strip() != "") &
            (combined["source"].str.strip() != "")
        ]
        after_filter = len(combined)
        
        if before_filter > after_filter:
            LOGGER.warning(
                f"Removed {before_filter - after_filter} rows with empty "
                f"question or source"
            )
        
        # Merge duplicate pairs (preserves many-to-many relationships)
        LOGGER.info("Merging duplicate pairs (same question + same source)")
        before_dedup = len(combined)
        merged = self._merge_duplicates(combined)
        after_dedup = len(merged)
        
        if before_dedup > after_dedup:
            LOGGER.info(
                f"Merged {before_dedup - after_dedup} duplicate pairs. "
                f"Result: {after_dedup} unique pairs"
            )
        else:
            LOGGER.info(f"No duplicate pairs found. Result: {after_dedup} unique pairs")
        
        # Assign IDs
        LOGGER.info("Assigning IDs")
        merged = self._assign_ids(merged)
        
        # Log statistics (showing many-to-many relationship)
        n_questions = merged["question_id"].nunique()
        n_sources = merged["source_id"].nunique()
        n_pairs = len(merged)
        
        LOGGER.info(f"Final statistics (many-to-many relationship):")
        LOGGER.info(f"  Questions: {n_questions}")
        LOGGER.info(f"  Sources: {n_sources}")
        LOGGER.info(f"  Pairs: {n_pairs}")
        LOGGER.info(f"  Avg sources per question: {n_pairs / n_questions:.2f}")
        LOGGER.info(f"  Avg questions per source: {n_pairs / n_sources:.2f}")
        
        # Save if output path provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(output_path, index=False)
            LOGGER.info(f"Saved merged DataFrame to {output_path}")
        
        return merged
    
    def create_inspection_metadata(
        self,
        df: pd.DataFrame,
        clustering_source: str
    ) -> Dict:
        """Create optional inspection.json metadata for debugging.
        
        Parameters
        ----------
        df : pd.DataFrame
            Merged DataFrame with IDs
        clustering_source : str
            Path to KeywordClusterer JSON
        
        Returns
        -------
        Dict
            Inspection metadata dictionary
        """
        import datetime
        
        # Extract unique questions
        question_data = (
            df[["question_id", "question"]]
            .drop_duplicates()
            .sort_values("question_id")
        )
        questions = [
            {"id": int(row["question_id"]), "text": row["question"]}
            for _, row in question_data.iterrows()
        ]
        
        # Extract unique sources
        source_data = (
            df[["source_id", "source"]]
            .drop_duplicates()
            .sort_values("source_id")
        )
        sources = [
            {"id": int(row["source_id"]), "text": row["source"]}
            for _, row in source_data.iterrows()
        ]
        
        # Metadata
        metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "n_pairs": len(df),
            "n_questions": len(questions),
            "n_sources": len(sources),
            "clustering_source": clustering_source,
            "keywords_present": any(df["keywords"].apply(len) > 0)
        }
        
        return {
            "metadata": metadata,
            "questions": questions[:100],  # Limit for file size
            "sources": sources[:100],  # Limit for file size
            "pair_samples": []  # Will be populated in later pipeline stages
        }


def merge_csv_files(
    csv_paths: List[Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None,
    column_aliases: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """Convenience function to merge CSV files.
    
    Preserves many-to-many relationships between questions and sources.
    Only exact duplicate pairs (same question + same source) are merged.
    
    Parameters
    ----------
    csv_paths : List[str or Path]
        Paths to CSV files to merge
    output_path : str or Path, optional
        If provided, save merged DataFrame to this path
    column_aliases : Dict[str, List[str]], optional
        Custom column name aliases
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with many-to-many relationships preserved
    
    Examples
    --------
    >>> df = merge_csv_files(
    ...     csv_paths=["data1.csv", "data2.csv"],
    ...     output_path="merged.csv"
    ... )
    >>> # Verify many-to-many relationship
    >>> print(f"Avg sources per question: {len(df) / df['question_id'].nunique():.2f}")
    """
    merger = CSVMerger(column_aliases=column_aliases)
    return merger.merge_csv_files(csv_paths=csv_paths, output_path=output_path)
