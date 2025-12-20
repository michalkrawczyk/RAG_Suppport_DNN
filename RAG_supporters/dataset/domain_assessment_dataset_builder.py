"""Dataset builder for domain assessment CSV + clustering JSON approach."""

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
from tqdm import tqdm

from clustering.clustering_data import ClusteringData
from domain_assessment_parser import DomainAssessmentParser
from label_calculator import LabelCalculator
from sqlite_storage import SQLiteStorageManager
from steering.steering_config import SteeringConfig
from steering_embedding_generator import SteeringEmbeddingGenerator


class DomainAssessmentDatasetBuilder:
    """
    Builds cluster-labeled dataset from CSV domain assessments + clustering JSON.

    Storage: SQLite (metadata/labels) + numpy memmap (embeddings).
    Labels: 3 types (source, steering, combined) for augmentation support.
    """

    def __init__(
        self,
        csv_paths: Union[str, Path, List[Union[str, Path]]],
        clustering_json_path: Union[str, Path],
        output_dir: Union[str, Path],
        embedding_model: Any,  # Model with encode() method
        steering_config: Optional[SteeringConfig] = None,
        label_normalizer: str = "softmax",
        label_temp: float = 1.0,
        combined_label_weight: float = 0.5,
        augment_noise_prob: float = 0.0,
        augment_zero_prob: float = 0.0,
        chunk_size: Optional[int] = None,
    ):
        """
        Initialize dataset builder.

        Args:
            csv_paths: Path(s) to CSV files from domain_assessment.py
            clustering_json_path: Path to clustering JSON from keyword_clustering.py
            output_dir: Output directory for dataset
            embedding_model: Model for encoding text
            steering_config: Steering configuration (defaults to ZERO mode)
            label_normalizer: Normalization method ('softmax', 'l1')
            label_temp: Temperature for softmax normalization
            combined_label_weight: Weight for combined labels (0=source, 1=steering)
            augment_noise_prob: Probability of noise augmentation
            augment_zero_prob: Probability of zero steering augmentation
            chunk_size: Chunk size for CSV reading (None = read all)
        """
        self.csv_paths = csv_paths if isinstance(csv_paths, list) else [csv_paths]
        self.clustering_json_path = Path(clustering_json_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size

        # Load clustering data
        self.clustering_data = ClusteringData.from_json(
            self.clustering_json_path,
            exclude_keys={"embeddings"},  # Don't load large embedding arrays
        )

        # Initialize steering config
        from RAG_supporters.dataset.steering.steering_config import SteeringMode

        self.steering_config = steering_config or SteeringConfig(
            mode=[(SteeringMode.ZERO, 1.0)]
        )

        # Initialize label calculator
        self.label_calculator = LabelCalculator(
            clustering_data=self.clustering_data,
            normalizer=label_normalizer,
            temperature=label_temp,
        )

        # Initialize steering generator
        self.steering_generator = SteeringEmbeddingGenerator(
            config=self.steering_config,
            clustering_data=self.clustering_data,
            embedding_model=embedding_model,
            augment_noise_prob=augment_noise_prob,
            augment_zero_prob=augment_zero_prob,
        )

        self.combined_label_weight = combined_label_weight

        # Initialize storage
        db_path = self.output_dir / "dataset.db"
        self.storage = SQLiteStorageManager(db_path)

        logging.info(f"Initialized DomainAssessmentDatasetBuilder")
        logging.info(f"  Output: {self.output_dir}")
        logging.info(f"  Clusters: {self.clustering_data.n_clusters}")
        logging.info(
            f"  Steering modes: {[m.value for m, _ in self.steering_config.mode]}"
        )

    def build(self) -> None:
        """Build complete dataset from CSV + clustering data."""
        logging.info("Starting dataset build...")

        # Parse CSV files
        parser = DomainAssessmentParser(chunk_size=self.chunk_size)
        data = parser.parse(self.csv_paths)

        logging.info(f"Parsed {len(data)} samples from CSV")

        # Prepare embedding arrays
        base_embeddings = []
        steering_embeddings = []

        # Process each sample
        for idx, sample in enumerate(tqdm(data, desc="Processing samples")):
            # Generate base embedding (source or question)
            text = sample.get("source_text") or sample.get("question_text")
            sample_type = "source" if sample.get("source_text") else "question"

            try:
                base_emb = self.embedding_model.encode([text])[0]
            except Exception as e:
                logging.error(f"Failed to encode sample {idx}: {e}")
                continue

            # Generate steering embedding
            steering_emb, steering_mode = self.steering_generator.generate(
                sample_id=idx,
                suggestions=sample.get("suggestions"),
                cluster_id=self._get_primary_cluster(sample),
            )

            # Calculate labels
            source_label = self._calculate_source_label(sample, base_emb)
            steering_label = self.label_calculator.calculate_steering_labels(
                steering_emb, self.clustering_data
            )
            combined_label = self.label_calculator.calculate_combined_labels(
                source_label, steering_label, self.combined_label_weight
            )

            # Store in database
            self.storage.insert_sample(
                sample_type=sample_type,
                text=text,
                source_label=source_label,
                steering_label=steering_label,
                combined_label=combined_label,
                embedding_idx=idx,
                chroma_id=sample.get("chroma_id"),
                suggestions=sample.get("suggestions"),
                steering_mode=steering_mode.value,
            )

            base_embeddings.append(base_emb)
            steering_embeddings.append(steering_emb)

        # Save embeddings as memmap files
        self._save_embeddings("base", np.array(base_embeddings, dtype=np.float32))
        self._save_embeddings(
            "steering", np.array(steering_embeddings, dtype=np.float32)
        )

        # Store dataset metadata
        self.storage.set_dataset_info("n_clusters", self.clustering_data.n_clusters)
        self.storage.set_dataset_info(
            "clustering_json_path", str(self.clustering_json_path)
        )
        self.storage.set_dataset_info(
            "label_normalizer", self.label_calculator.normalizer
        )
        self.storage.set_dataset_info(
            "combined_label_weight", self.combined_label_weight
        )

        logging.info(f"Dataset build complete: {len(data)} samples")
        logging.info(f"  Database: {self.storage.db_path}")
        logging.info(f"  Base embeddings: {self.output_dir / 'base_embeddings.npy'}")
        logging.info(
            f"  Steering embeddings: {self.output_dir / 'steering_embeddings.npy'}"
        )

    def _calculate_source_label(
        self, sample: dict, base_embedding: np.ndarray
    ) -> np.ndarray:
        """Calculate source/question label from CSV probabilities or embedding."""
        # Try CSV probabilities first
        if "cluster_probabilities" in sample and sample["cluster_probabilities"]:
            probs = sample["cluster_probabilities"]
            if len(probs) == self.clustering_data.n_clusters:
                return np.array(probs, dtype=np.float32)

        # Fallback: use suggestions
        if "suggestions" in sample and sample["suggestions"]:
            return self.label_calculator.calculate_source_labels(
                sample["suggestions"], self.clustering_data, self.embedding_model
            )

        # Last resort: use embedding distance
        return self.label_calculator.calculate_question_labels(
            base_embedding, None, self.clustering_data
        )

    def _get_primary_cluster(self, sample: dict) -> Optional[int]:
        """Get primary cluster ID from sample."""
        if "cluster_probabilities" in sample and sample["cluster_probabilities"]:
            probs = sample["cluster_probabilities"]
            return int(np.argmax(probs))
        return None

    def _save_embeddings(self, embedding_type: str, embeddings: np.ndarray):
        """Save embeddings as memory-mapped numpy array."""
        file_path = self.output_dir / f"{embedding_type}_embeddings.npy"
        np.save(str(file_path), embeddings)

        self.storage.register_embedding_file(
            embedding_type=embedding_type,
            file_path=file_path,
            shape=embeddings.shape,
            dtype=str(embeddings.dtype),
        )

        logging.info(f"Saved {embedding_type} embeddings: shape={embeddings.shape}")

    def close(self):
        """Close storage connections."""
        self.storage.close()
