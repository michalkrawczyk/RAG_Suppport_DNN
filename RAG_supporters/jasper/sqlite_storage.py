"""SQLite storage manager for domain assessment dataset."""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class SQLiteStorageManager:
    """
    Manages SQLite database for dataset metadata, labels, and text fields.

    Schema:
    - samples: Main table with metadata, labels, text
    - embeddings_meta: Tracks numpy memmap files for embeddings
    - dataset_info: Dataset-level metadata
    """

    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize SQLite storage manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Initialize database schema if not exists."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable column access by name

        # Create tables
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_type TEXT NOT NULL,  -- 'source' or 'question'
                text TEXT NOT NULL,
                chroma_id TEXT,
                suggestions TEXT,  -- JSON list of suggestions
                source_label TEXT NOT NULL,  -- JSON array of probabilities
                steering_label TEXT NOT NULL,  -- JSON array of probabilities
                combined_label TEXT NOT NULL,  -- JSON array of probabilities
                embedding_idx INTEGER NOT NULL,  -- Index into memmap file
                steering_mode TEXT,  -- Mode used for steering embedding
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings_meta (
                embedding_type TEXT PRIMARY KEY,  -- 'base' or 'steering'
                file_path TEXT NOT NULL,
                shape TEXT NOT NULL,  -- JSON [n_samples, embedding_dim]
                dtype TEXT NOT NULL,
                checksum TEXT NOT NULL  -- SHA256 of file
            )
        """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_info (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL  -- JSON-encoded value
            )
        """
        )

        # Create indices for faster queries
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sample_type 
            ON samples(sample_type)
        """
        )

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chroma_id 
            ON samples(chroma_id)
        """
        )

        self.conn.commit()
        logging.info(f"Initialized SQLite database at {self.db_path}")

    def insert_sample(
        self,
        sample_type: str,
        text: str,
        source_label: np.ndarray,
        steering_label: np.ndarray,
        combined_label: np.ndarray,
        embedding_idx: int,
        chroma_id: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        steering_mode: Optional[str] = None,
    ) -> int:
        """
        Insert a sample into the database.

        Args:
            sample_type: 'source' or 'question'
            text: Source or question text
            source_label: Label for source/question (numpy array)
            steering_label: Label for steering embedding (numpy array)
            combined_label: Combined label (numpy array)
            embedding_idx: Index into memmap file
            chroma_id: ChromaDB ID
            suggestions: List of suggestion strings
            steering_mode: Steering mode used

        Returns:
            sample_id of inserted row
        """
        cursor = self.conn.execute(
            """
            INSERT INTO samples (
                sample_type, text, chroma_id, suggestions,
                source_label, steering_label, combined_label,
                embedding_idx, steering_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample_type,
                text,
                chroma_id,
                json.dumps(suggestions) if suggestions else None,
                json.dumps(source_label.tolist()),
                json.dumps(steering_label.tolist()),
                json.dumps(combined_label.tolist()),
                embedding_idx,
                steering_mode,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_sample(self, sample_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a sample by ID.

        Args:
            sample_id: Sample ID

        Returns:
            Dictionary with sample data or None
        """
        cursor = self.conn.execute("SELECT * FROM samples WHERE sample_id = ?", (sample_id,))
        row = cursor.fetchone()
        if row is None:
            return None

        return self._row_to_dict(row)

    def get_all_samples(self) -> List[Dict[str, Any]]:
        """
        Retrieve all samples.

        Returns:
            List of sample dictionaries
        """
        cursor = self.conn.execute("SELECT * FROM samples ORDER BY sample_id")
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_dataset_size(self) -> int:
        """
        Get total number of samples in dataset.

        Returns:
            Number of samples
        """
        cursor = self.conn.execute("SELECT COUNT(*) FROM samples")
        return cursor.fetchone()[0]

    def get_sample_by_index(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve sample by its index (0-based position in ordered dataset).

        Args:
            idx: Sample index (0-based)

        Returns:
            Sample dictionary or None if index out of range
        """
        cursor = self.conn.execute(
            "SELECT * FROM samples ORDER BY sample_id LIMIT 1 OFFSET ?", (idx,)
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return self._row_to_dict(row)

    def update_labels(
        self,
        sample_id: int,
        source_label: Optional[np.ndarray] = None,
        steering_label: Optional[np.ndarray] = None,
        combined_label: Optional[np.ndarray] = None,
    ):
        """
        Update labels for a sample.

        Args:
            sample_id: Sample ID
            source_label: New source label (if provided)
            steering_label: New steering label (if provided)
            combined_label: New combined label (if provided)
        """
        updates = []
        params = []

        if source_label is not None:
            updates.append("source_label = ?")
            params.append(json.dumps(source_label.tolist()))

        if steering_label is not None:
            updates.append("steering_label = ?")
            params.append(json.dumps(steering_label.tolist()))

        if combined_label is not None:
            updates.append("combined_label = ?")
            params.append(json.dumps(combined_label.tolist()))

        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(sample_id)

            query = f"UPDATE samples SET {', '.join(updates)} WHERE sample_id = ?"
            self.conn.execute(query, params)
            self.conn.commit()

    def register_embedding_file(
        self,
        embedding_type: str,
        file_path: Union[str, Path],
        shape: Tuple[int, int],
        dtype: str,
    ):
        """
        Register a numpy memmap file for embeddings.

        Args:
            embedding_type: 'base' or 'steering'
            file_path: Path to .npy file
            shape: Shape tuple (n_samples, embedding_dim)
            dtype: Numpy dtype string
        """
        file_path = Path(file_path)
        checksum = self._compute_file_checksum(file_path)

        self.conn.execute(
            """
            INSERT OR REPLACE INTO embeddings_meta 
            (embedding_type, file_path, shape, dtype, checksum)
            VALUES (?, ?, ?, ?, ?)
            """,
            (embedding_type, str(file_path), json.dumps(shape), dtype, checksum),
        )
        self.conn.commit()
        logging.info(f"Registered {embedding_type} embeddings: {file_path}")

    def get_embedding_file_info(self, embedding_type: str) -> Optional[Dict[str, Any]]:
        """
        Get info about a registered embedding file.

        Args:
            embedding_type: 'base' or 'steering'

        Returns:
            Dictionary with file info or None
        """
        cursor = self.conn.execute(
            "SELECT * FROM embeddings_meta WHERE embedding_type = ?", (embedding_type,)
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return {
            "embedding_type": row["embedding_type"],
            "file_path": row["file_path"],
            "shape": json.loads(row["shape"]),
            "dtype": row["dtype"],
            "checksum": row["checksum"],
        }

    def set_dataset_info(self, key: str, value: Any):
        """
        Store dataset-level metadata.

        Args:
            key: Metadata key
            value: Value (will be JSON-encoded)
        """
        self.conn.execute(
            "INSERT OR REPLACE INTO dataset_info (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def get_dataset_info(self, key: str) -> Optional[Any]:
        """
        Retrieve dataset-level metadata.

        Args:
            key: Metadata key

        Returns:
            Value (JSON-decoded) or None
        """
        cursor = self.conn.execute("SELECT value FROM dataset_info WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row["value"])

    def get_sample_count(self) -> int:
        """Get total number of samples."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM samples")
        return cursor.fetchone()[0]

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary with parsed JSON fields."""
        return {
            "sample_id": row["sample_id"],
            "sample_type": row["sample_type"],
            "text": row["text"],
            "chroma_id": row["chroma_id"],
            "suggestions": (json.loads(row["suggestions"]) if row["suggestions"] else None),
            "source_label": np.array(json.loads(row["source_label"]), dtype=np.float32),
            "steering_label": np.array(json.loads(row["steering_label"]), dtype=np.float32),
            "combined_label": np.array(json.loads(row["combined_label"]), dtype=np.float32),
            "embedding_idx": row["embedding_idx"],
            "steering_mode": row["steering_mode"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _compute_file_checksum(file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close connection."""
        self.close()
