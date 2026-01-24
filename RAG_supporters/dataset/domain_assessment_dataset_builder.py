"""Dataset builder for domain assessment CSV + clustering JSON approach."""

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
from RAG_supporters.clustering.clustering_data import ClusteringData
from RAG_supporters.embeddings.keyword_embedder import KeywordEmbedder
from tqdm import tqdm

from .domain_assessment_parser import DomainAssessmentParser
from .label_calculator import LabelCalculator, LabelNormalizationMethod
from .sqlite_storage import SQLiteStorageManager
from .steering.steering_config import SteeringConfig, SteeringMode
from .steering_embedding_generator import SteeringEmbeddingGenerator


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
        embedding_model: Union[
            str, Any, KeywordEmbedder
        ],  # Model name, KeywordEmbedder, or raw model
        steering_config: Optional[SteeringConfig] = None,
        label_normalizer: str = "softmax",
        label_temp: float = 1.0,
        combined_label_weight: float = 0.5,
        augment_noise_prob: float = 0.0,
        augment_zero_prob: float = 0.0,
        augment_noise_level: float = 0.01,
        chunk_size: Optional[int] = None,
        source_suggestions_col: str = "suggestions",
        question_suggestions_col: str = "selected_terms",
    ):
        """
        Initialize dataset builder.

        Args:
            csv_paths: Path(s) to CSV files from domain_assessment.py
            clustering_json_path: Path to clustering JSON from keyword_clustering.py
            output_dir: Output directory for dataset
            embedding_model: Model name (str), KeywordEmbedder instance, or raw model.
                           If string: creates KeywordEmbedder with model_name.
                           Supports both sentence-transformers and LangChain models.
            steering_config: Steering configuration (defaults to ZERO mode)
            label_normalizer: Normalization method ('softmax', 'l1')
            label_temp: Temperature for softmax normalization
            combined_label_weight: Weight for combined labels (0=source, 1=steering)
            augment_noise_prob: Probability of noise augmentation
            augment_zero_prob: Probability of zero steering augmentation
            augment_noise_level: Std deviation for noise augmentation
            chunk_size: Chunk size for CSV reading (None = read all)
            source_suggestions_col: Column name for source suggestions (default: 'suggestions')
            question_suggestions_col: Column name for question suggestions (default: 'selected_terms')
        """
        self.csv_paths = csv_paths if isinstance(csv_paths, list) else [csv_paths]
        self.clustering_json_path = Path(clustering_json_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.source_suggestions_col = source_suggestions_col
        self.question_suggestions_col = question_suggestions_col

        # Initialize KeywordEmbedder (supports both sentence-transformers and LangChain)
        if isinstance(embedding_model, KeywordEmbedder):
            self.embedder = embedding_model
        elif isinstance(embedding_model, str):
            # Create KeywordEmbedder from model name
            self.embedder = KeywordEmbedder(model_name=embedding_model)
        else:
            self.embedder = KeywordEmbedder(embedding_model=embedding_model)

        self.chunk_size = chunk_size

        # Load clustering data (including suggestion embeddings)
        self.clustering_data = ClusteringData.from_json(
            self.clustering_json_path,
            exclude_keys=set(),  # Load all data including embeddings
        )

        # Load suggestion embeddings from clustering JSON
        self.suggestion_embeddings = self._load_suggestion_embeddings()

        # Initialize steering config
        self.steering_config = steering_config or SteeringConfig(
            mode=[(SteeringMode.ZERO, 1.0)]
        )

        # Initialize label calculator
        self.label_calculator = LabelCalculator(
            clustering_data=self.clustering_data,
            normalization_method=LabelNormalizationMethod(label_normalizer),
            temperature=label_temp,
        )

        # Initialize steering generator
        self.steering_generator = SteeringEmbeddingGenerator(
            config=self.steering_config,
            clustering_data=self.clustering_data,
            embedding_model=self.embedder,
            augment_noise_prob=augment_noise_prob,
            augment_zero_prob=augment_zero_prob,
            augment_noise_level=augment_noise_level,
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
        parser = DomainAssessmentParser(
            chunksize=self.chunk_size,
            source_suggestions_col=self.source_suggestions_col,
            question_suggestions_col=self.question_suggestions_col,
        )
        if isinstance(self.csv_paths, (str, Path)):
            data = parser.parse_csv(self.csv_paths)
        else:
            data = (
                parser.parse_csv(self.csv_paths[0])
                if len(self.csv_paths) == 1
                else parser.parse_multiple_csvs(self.csv_paths)
            )

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
                base_emb = self.embedder._generate_embeddings(
                    [text], batch_size=1, show_progress=False
                )[0]
            except Exception as e:
                logging.error(f"Failed to encode sample {idx}: {e}")
                continue

            # Generate steering embedding
            steering_emb, steering_mode = self.steering_generator.generate(
                sample_id=idx,
                suggestions=sample.get(
                    self.source_suggestions_col
                    if sample_type == "source"
                    else self.question_suggestions_col
                ),
                cluster_id=self._get_primary_cluster(sample, sample_type),
            )

            # Calculate labels
            source_label = self._calculate_source_label(sample, base_emb, sample_type)
            steering_label = self.label_calculator.calculate_steering_labels(
                steering_emb
            )
            combined_label = self.label_calculator.calculate_combined_labels(
                source_label, steering_label, self.combined_label_weight
            )

            # Get correct chroma_id based on sample type
            chroma_id = (
                sample.get("chroma_source_id")
                if sample_type == "source"
                else sample.get("chroma_question_id")
            )

            # Store in database
            self.storage.insert_sample(
                sample_type=sample_type,
                text=text,
                source_label=source_label,
                steering_label=steering_label,
                combined_label=combined_label,
                embedding_idx=idx,
                chroma_id=chroma_id,
                suggestions=sample.get(
                    self.source_suggestions_col
                    if sample_type == "source"
                    else self.question_suggestions_col
                ),
                steering_mode=steering_mode.value,
            )

            base_embeddings.append(base_emb)
            steering_embeddings.append(steering_emb)

        # Save embeddings as memmap files
        base_embeddings_array = np.array(base_embeddings, dtype=np.float32)
        steering_embeddings_array = np.array(steering_embeddings, dtype=np.float32)

        self._save_embeddings("base", base_embeddings_array)
        self._save_embeddings("steering", steering_embeddings_array)

        # Store dataset metadata
        self.storage.set_dataset_info("n_clusters", self.clustering_data.n_clusters)
        self.storage.set_dataset_info(
            "embedding_dim", int(base_embeddings_array.shape[1])
        )
        self.storage.set_dataset_info(
            "clustering_json_path", str(self.clustering_json_path)
        )
        self.storage.set_dataset_info(
            "label_normalizer", type(self.label_calculator.normalizer).__name__
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
        self, sample: dict, base_embedding: np.ndarray, sample_type: str
    ) -> np.ndarray:
        """Calculate source/question label from CSV probabilities or embedding.

        Args:
            sample: Sample dictionary from CSV
            base_embedding: Embedding of the text
            sample_type: Type of sample ('source' or 'question') - used to select appropriate column
        """
        # Try CSV probabilities first
        if "cluster_probabilities" in sample and sample["cluster_probabilities"]:
            probs = sample["cluster_probabilities"]
            if len(probs) == self.clustering_data.n_clusters:
                return np.array(probs, dtype=np.float32)

        # Fallback: use suggestions from appropriate column
        suggestions = sample.get(
            self.source_suggestions_col
            if sample_type == "source"
            else self.question_suggestions_col
        )
        if suggestions:
            return self.label_calculator.calculate_source_labels_from_suggestions(
                suggestions, self.suggestion_embeddings
            )

        # Last resort: use embedding distance
        return self.label_calculator.calculate_question_labels_from_embedding(
            base_embedding
        )

    def _get_primary_cluster(self, sample: dict, sample_type: str) -> Optional[int]:
        """Get primary cluster ID from sample.

        Args:
            sample: Sample dictionary from CSV
            sample_type: Type of sample ('source' or 'question') - used to select appropriate column
        """
        # Try CSV probabilities first
        if "cluster_probabilities" in sample and sample["cluster_probabilities"]:
            probs = sample["cluster_probabilities"]
            return int(np.argmax(probs))

        # Fallback: calculate from suggestions using appropriate column based on sample_type
        suggestions = sample.get(
            self.source_suggestions_col
            if sample_type == "source"
            else self.question_suggestions_col
        )
        if suggestions:
            try:
                source_label = (
                    self.label_calculator.calculate_source_labels_from_suggestions(
                        suggestions, self.suggestion_embeddings
                    )
                )
                primary_cluster = int(np.argmax(source_label))
                logging.debug(
                    f"Calculated primary cluster {primary_cluster} from {len(suggestions)} suggestions"
                )
                return primary_cluster
            except Exception as e:
                logging.warning(
                    f"Failed to calculate primary cluster from suggestions: {e}"
                )
                return None

        # No way to determine cluster
        logging.debug(
            "No cluster_probabilities or suggestions available, returning None"
        )
        return None

    def _load_suggestion_embeddings(self) -> dict:
        """
        Load suggestion/keyword embeddings from clustering JSON.

        Returns:
            Dictionary mapping suggestion terms to embeddings
        """
        import json

        with open(self.clustering_json_path, "r") as f:
            clustering_json = json.load(f)

        # Get embeddings from JSON (saved by keyword_clustering with include_embeddings=True)
        embeddings_dict = clustering_json.get("embeddings", {})

        if not embeddings_dict:
            logging.warning(
                f"No embeddings found in {self.clustering_json_path}. "
                "Suggestion-based label calculation will fall back to uniform distribution."
            )
            return {}

        # Convert lists back to numpy arrays
        suggestion_embeddings = {
            term: np.array(emb, dtype=np.float32)
            for term, emb in embeddings_dict.items()
        }

        logging.info(f"Loaded {len(suggestion_embeddings)} suggestion embeddings")
        return suggestion_embeddings

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
