import json
import logging
import pickle
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from langchain_core.embeddings.embeddings import Embeddings
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils.dataset_loader import (
    compute_cache_version,
    filter_suggestions,
    parse_suggestions_safe,
)


class SteeringMode(Enum):
    """Steering embedding modes for cluster/subspace steering."""

    SUGGESTION = "suggestion"  # Use suggestion embeddings
    LLM_GENERATED = "llm_generated"  # LLM-generated steering text
    CLUSTER_DESCRIPTOR = "cluster_descriptor"  # Cluster/topic descriptor embeddings
    ZERO = "zero"  # Zero baseline (no steering)
    MIXED = "mixed"  # Weighted combination of multiple modes


class BaseDomainAssignDataset(Dataset):
    """
    PyTorch Dataset for processing raw data with on-the-fly or batch embedding computation.
    Can be used directly as a dataset OR to build a cached version.

    Extended to support flexible cluster steering with multiple steering embedding modes.

    Usage as Dataset (no caching):
        builder = DomainAssignDatasetBuilder(
            df=df,
            embedding_model=embeddings,
            return_embeddings=True
        )
        sample = builder[0]  # Computes embeddings on-the-fly

    Usage as Builder (with caching):
        builder = DomainAssignDatasetBuilder(
            df=df,
            embedding_model=embeddings,
            cache_dir='./cache'
        )
        builder.build()  # Precomputes and saves everything

        # Later load with DomainAssignDataset
        dataset = DomainAssignDataset(cache_dir='./cache')

    Cluster Steering Usage:
        builder = DomainAssignDatasetBuilder(
            df=df,
            embedding_model=embeddings,
            steering_mode=SteeringMode.CLUSTER_DESCRIPTOR,
            cluster_labels=cluster_assignments,  # Dict[int, List[int]] or soft labels
            cluster_descriptors=topic_descriptors,  # Dict[int, List[str]]
            return_triplets=True  # Returns (base, steering, target)
        )
    """

    def __init__(
        self,
        df: Union[pd.DataFrame, str, Path],
        embedding_model: Optional[Embeddings] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        source_col: str = "source",
        question_col: str = "question",
        suggestions_col: str = "suggestions",
        min_confidence: float = 0.0,
        suggestion_types: Optional[List[str]] = None,
        return_embeddings: bool = True,
        chunksize: int = 10000,
        embedding_model_name: Optional[str] = None,
        # Cluster steering parameters
        steering_mode: Optional[Union[SteeringMode, str]] = None,
        cluster_labels: Optional[Union[Dict[int, int], Dict[int, List[float]]]] = None,
        cluster_descriptors: Optional[Dict[int, List[str]]] = None,
        llm_steering_texts: Optional[Dict[int, str]] = None,
        return_triplets: bool = False,
        multi_label_mode: str = "hard",  # "hard" or "soft"
        steering_weights: Optional[Dict[str, float]] = None,
    ):
        self.source_col = source_col
        self.question_col = question_col
        self.suggestions_col = suggestions_col
        self.min_confidence = min_confidence
        self.suggestion_types = suggestion_types
        self.embedding_model = embedding_model
        self.return_embeddings = return_embeddings and embedding_model is not None
        self.chunksize = chunksize
        self.embedding_model_name = embedding_model_name or "unknown"

        # Cluster steering parameters
        if isinstance(steering_mode, str):
            steering_mode = SteeringMode(steering_mode)
        self.steering_mode = steering_mode
        self.cluster_labels = cluster_labels
        self.cluster_descriptors = cluster_descriptors
        self.llm_steering_texts = llm_steering_texts
        self.return_triplets = return_triplets
        self.multi_label_mode = multi_label_mode
        self.steering_weights = steering_weights or {}

        # Validate
        if self.return_embeddings and not embedding_model:
            raise ValueError(
                "embedding_model must be provided when return_embeddings=True"
            )
        
        if self.return_triplets and not self.steering_mode:
            raise ValueError(
                "steering_mode must be specified when return_triplets=True"
            )

        # Setup cache directory (optional)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        # TODO: Shoud'nt be read in chunks here?
        if isinstance(df, (str, Path)):
            logging.info(f"Loading CSV from {df}")
            self.df = pd.read_csv(df)
        else:
            self.df = df.reset_index(drop=True)

        logging.info(f"Loaded {len(self.df)} rows")

        # Compute version for cache validation
        self.version = compute_cache_version(
            min_confidence, suggestion_types, self.embedding_model_name
        )

        # In-memory caches (built on-demand or via build())
        self._parsed_suggestions_cache: Optional[List[List[str]]] = None
        self._unique_suggestions_cache: Optional[List[str]] = None
        self._text_embeddings_cache: Optional[List[Dict[str, np.ndarray]]] = None
        self._suggestion_embeddings_cache: Optional[Dict[str, np.ndarray]] = None
        
        # Cluster steering caches
        self._cluster_descriptor_embeddings_cache: Optional[Dict[int, np.ndarray]] = None
        self._llm_steering_embeddings_cache: Optional[Dict[int, np.ndarray]] = None
        self._steering_embeddings_cache: Optional[List[np.ndarray]] = None

    def __len__(self) -> int:
        return len(self.df)

    def _generate_steering_embedding(
        self, idx: int, suggestions: List[str]
    ) -> Optional[np.ndarray]:
        """
        Generate steering embedding based on configured steering mode.

        Args:
            idx: Sample index
            suggestions: List of suggestion texts

        Returns:
            Steering embedding as numpy array or None
        """
        if not self.steering_mode:
            return None

        embedding_dim = (
            self._text_embeddings_cache[0]["question"].shape[0]
            if self._text_embeddings_cache
            else None
        )

        if self.steering_mode == SteeringMode.ZERO:
            # Zero baseline - return zero vector
            if embedding_dim is None:
                raise ValueError("Cannot determine embedding dimension for zero vector")
            return np.zeros(embedding_dim, dtype=np.float32)

        elif self.steering_mode == SteeringMode.SUGGESTION:
            # Use first suggestion embedding as steering
            if not suggestions:
                if embedding_dim:
                    return np.zeros(embedding_dim, dtype=np.float32)
                return None
            
            # Get suggestion embedding
            if self._steering_embeddings_cache is not None:
                return self._steering_embeddings_cache[idx]
            
            # Compute on-the-fly
            if self._suggestion_embeddings_cache and suggestions[0] in self._suggestion_embeddings_cache:
                return self._suggestion_embeddings_cache[suggestions[0]]
            
            if self.embedding_model:
                emb = np.array(
                    self.embedding_model.embed_query(suggestions[0]), dtype=np.float32
                )
                return emb
            return None

        elif self.steering_mode == SteeringMode.CLUSTER_DESCRIPTOR:
            # Use cluster descriptor embedding
            if not self.cluster_labels or not self.cluster_descriptors:
                if embedding_dim:
                    return np.zeros(embedding_dim, dtype=np.float32)
                return None

            # Get cluster assignment for this sample
            cluster_id = self._get_primary_cluster(idx)
            if cluster_id is None:
                if embedding_dim:
                    return np.zeros(embedding_dim, dtype=np.float32)
                return None

            # Get cached descriptor embedding
            if (
                self._cluster_descriptor_embeddings_cache
                and cluster_id in self._cluster_descriptor_embeddings_cache
            ):
                return self._cluster_descriptor_embeddings_cache[cluster_id]

            # Compute on-the-fly
            if cluster_id in self.cluster_descriptors and self.embedding_model:
                descriptors = self.cluster_descriptors[cluster_id]
                if descriptors:
                    # Use first descriptor
                    emb = np.array(
                        self.embedding_model.embed_query(descriptors[0]),
                        dtype=np.float32,
                    )
                    return emb
            
            if embedding_dim:
                return np.zeros(embedding_dim, dtype=np.float32)
            return None

        elif self.steering_mode == SteeringMode.LLM_GENERATED:
            # Use LLM-generated steering text
            if not self.llm_steering_texts:
                if embedding_dim:
                    return np.zeros(embedding_dim, dtype=np.float32)
                return None

            # Get cached LLM steering embedding
            if (
                self._llm_steering_embeddings_cache is not None
                and idx in self._llm_steering_embeddings_cache
            ):
                return self._llm_steering_embeddings_cache[idx]

            # Compute on-the-fly
            if idx in self.llm_steering_texts and self.embedding_model:
                steering_text = self.llm_steering_texts[idx]
                emb = np.array(
                    self.embedding_model.embed_query(steering_text), dtype=np.float32
                )
                return emb

            if embedding_dim:
                return np.zeros(embedding_dim, dtype=np.float32)
            return None

        elif self.steering_mode == SteeringMode.MIXED:
            # Weighted combination of multiple steering embeddings
            embeddings = []
            weights = []

            # Collect embeddings based on weights
            if self.steering_weights.get("suggestion", 0) > 0 and suggestions:
                if self._suggestion_embeddings_cache and suggestions[0] in self._suggestion_embeddings_cache:
                    embeddings.append(self._suggestion_embeddings_cache[suggestions[0]])
                    weights.append(self.steering_weights["suggestion"])

            if self.steering_weights.get("cluster_descriptor", 0) > 0:
                cluster_id = self._get_primary_cluster(idx)
                if (
                    cluster_id is not None
                    and self._cluster_descriptor_embeddings_cache
                    and cluster_id in self._cluster_descriptor_embeddings_cache
                ):
                    embeddings.append(
                        self._cluster_descriptor_embeddings_cache[cluster_id]
                    )
                    weights.append(self.steering_weights["cluster_descriptor"])

            if not embeddings:
                if embedding_dim:
                    return np.zeros(embedding_dim, dtype=np.float32)
                return None

            # Weighted average
            weights = np.array(weights, dtype=np.float32)
            weights = weights / weights.sum()  # Normalize
            mixed_emb = sum(w * emb for w, emb in zip(weights, embeddings))
            return mixed_emb.astype(np.float32)

        return None

    def _get_primary_cluster(self, idx: int) -> Optional[int]:
        """Get primary cluster assignment for a sample."""
        if not self.cluster_labels or idx not in self.cluster_labels:
            return None

        assignment = self.cluster_labels[idx]
        if isinstance(assignment, int):
            return assignment
        elif isinstance(assignment, list) and assignment:
            # For soft labels, return index with highest probability
            return int(np.argmax(assignment))
        return None

    def _get_num_clusters(self) -> int:
        """
        Determine the number of clusters from cluster_labels.
        
        Returns:
            Number of clusters
        """
        if not self.cluster_labels:
            raise ValueError("No cluster labels available")
        
        # Get first assignment to check format
        first_assignment = next(iter(self.cluster_labels.values()))
        
        if isinstance(first_assignment, int):
            # For int labels, find max + 1
            return max(self.cluster_labels.values()) + 1
        elif isinstance(first_assignment, list):
            # For probability distributions, use length
            return len(first_assignment)
        else:
            raise ValueError(f"Unexpected cluster label type: {type(first_assignment)}")

    def _generate_target(self, idx: int) -> Optional[Union[int, np.ndarray]]:
        """
        Generate target (cluster/subspace assignment) for a sample.

        Args:
            idx: Sample index

        Returns:
            Target as int (hard) or numpy array (soft) or None
        """
        if not self.cluster_labels or idx not in self.cluster_labels:
            return None

        assignment = self.cluster_labels[idx]

        if self.multi_label_mode == "hard":
            # Return cluster index
            if isinstance(assignment, int):
                return assignment
            elif isinstance(assignment, list):
                # Return primary cluster (argmax)
                return int(np.argmax(assignment))

        elif self.multi_label_mode == "soft":
            # Return probability distribution
            if isinstance(assignment, list):
                return np.array(assignment, dtype=np.float32)
            elif isinstance(assignment, int):
                # Convert int to one-hot distribution
                n_clusters = self._get_num_clusters()
                one_hot = np.zeros(n_clusters, dtype=np.float32)
                one_hot[assignment] = 1.0
                return one_hot

        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample, computing embeddings on-the-fly if needed.

        Returns:
            dict with keys (standard mode):
                - 'source': source text or embedding
                - 'question': question text or embedding
                - 'suggestions': list of suggestion terms or embeddings
                - 'suggestion_texts': list of suggestion terms (always included)
                - 'idx': index in dataset
            
            dict with keys (triplet mode, when return_triplets=True):
                - 'base_embedding': question embedding (source/query)
                - 'steering_embedding': steering embedding based on mode
                - 'target': cluster/subspace assignment (int or array)
                - 'metadata': dict with steering info, texts, cluster info
                - 'idx': index in dataset
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        row = self.df.iloc[idx]

        # Get parsed suggestions (use cache if available)
        if self._parsed_suggestions_cache is not None:
            suggestions = self._parsed_suggestions_cache[idx]
        else:
            raw_suggestions = row[self.suggestions_col]
            suggestions_parsed = parse_suggestions_safe(raw_suggestions)
            suggestions = filter_suggestions(
                suggestions_parsed, self.min_confidence, self.suggestion_types
            )

        if self.return_triplets and self.return_embeddings:
            # Triplet mode: return (base, steering, target, metadata)
            
            # Get base embedding (question)
            if self._text_embeddings_cache is not None:
                text_emb = self._text_embeddings_cache[idx]
                question_emb = text_emb["question"]
                source_emb = text_emb["source"]
            else:
                question_text = str(row[self.question_col])
                source_text = str(row[self.source_col])
                question_emb = np.array(
                    self.embedding_model.embed_query(question_text), dtype=np.float32
                )
                source_emb = np.array(
                    self.embedding_model.embed_query(source_text), dtype=np.float32
                )

            # Get steering embedding
            steering_emb = self._generate_steering_embedding(idx, suggestions)
            
            # Get target
            target = self._generate_target(idx)
            
            # Build metadata
            metadata = {
                "steering_mode": self.steering_mode.value if self.steering_mode else None,
                "suggestion_texts": suggestions,
                "source_text": str(row[self.source_col]),
                "question_text": str(row[self.question_col]),
                "sample_index": idx,
            }
            
            # Add cluster info if available
            if self.cluster_labels and idx in self.cluster_labels:
                metadata["cluster_assignment"] = self.cluster_labels[idx]
                cluster_id = self._get_primary_cluster(idx)
                if cluster_id is not None and self.cluster_descriptors and cluster_id in self.cluster_descriptors:
                    metadata["cluster_descriptors"] = self.cluster_descriptors[cluster_id]
            
            # Add LLM steering text if available
            if self.llm_steering_texts and idx in self.llm_steering_texts:
                metadata["llm_steering_text"] = self.llm_steering_texts[idx]

            result = {
                "base_embedding": torch.from_numpy(question_emb),
                "idx": idx,
                "metadata": metadata,
            }
            
            if steering_emb is not None:
                result["steering_embedding"] = torch.from_numpy(steering_emb)
            
            if target is not None:
                if isinstance(target, np.ndarray):
                    result["target"] = torch.from_numpy(target)
                else:
                    result["target"] = torch.tensor(target, dtype=torch.long)
            
            # Also include source for completeness
            result["source_embedding"] = torch.from_numpy(source_emb)
            
            return result

        elif self.return_embeddings:
            # Standard embedding mode
            # Get text embeddings
            if self._text_embeddings_cache is not None:
                # Use cached embeddings
                text_emb = self._text_embeddings_cache[idx]
                source_emb = text_emb["source"]
                question_emb = text_emb["question"]
            else:
                # Compute on-the-fly
                source_text = str(row[self.source_col])
                question_text = str(row[self.question_col])
                source_emb = np.array(
                    self.embedding_model.embed_query(source_text), dtype=np.float32
                )
                question_emb = np.array(
                    self.embedding_model.embed_query(question_text), dtype=np.float32
                )

            # Get suggestion embeddings
            suggestion_embeds = []
            for term in suggestions:
                if (
                    self._suggestion_embeddings_cache is not None
                    and term in self._suggestion_embeddings_cache
                ):
                    # Use cached
                    emb = self._suggestion_embeddings_cache[term]
                else:
                    # Compute on-the-fly
                    emb = np.array(
                        self.embedding_model.embed_query(term), dtype=np.float32
                    )
                    # Optionally cache it
                    if self._suggestion_embeddings_cache is not None:
                        self._suggestion_embeddings_cache[term] = emb

                suggestion_embeds.append(emb)

            return {
                "source": torch.from_numpy(source_emb),
                "question": torch.from_numpy(question_emb),
                "suggestions": [torch.from_numpy(emb) for emb in suggestion_embeds],
                "suggestion_texts": suggestions,
                "idx": idx,
            }
        else:
            # Return raw text
            return {
                "source": str(row[self.source_col]),
                "question": str(row[self.question_col]),
                "suggestions": suggestions,
                "suggestion_texts": suggestions,
                "idx": idx,
            }

    def build(self, save_to_cache: bool = True):
        """
        Build the complete dataset by precomputing all embeddings.

        Args:
            save_to_cache: If True and cache_dir is set, save to disk
        """
        total_steps = 6 if self.return_triplets else 4
        logging.info("=" * 60)
        logging.info("Building Domain Assign Dataset")
        if self.cache_dir:
            logging.info(f"Cache directory: {self.cache_dir}")
        logging.info(f"Version: {self.version}")
        if self.steering_mode:
            logging.info(f"Steering mode: {self.steering_mode.value}")
        logging.info("=" * 60)

        # Step 1: Parse and filter suggestions
        logging.info(f"\n[1/{total_steps}] Parsing suggestions...")
        self._parsed_suggestions_cache = self._parse_all_suggestions()

        # Step 2: Extract unique suggestions
        logging.info(f"\n[2/{total_steps}] Extracting unique suggestions...")
        self._unique_suggestions_cache = self._extract_unique_suggestions()

        # Step 3: Compute text embeddings (source + question)
        if self.return_embeddings:
            logging.info(f"\n[3/{total_steps}] Computing text embeddings...")
            self._text_embeddings_cache = self._compute_text_embeddings()

            # Step 4: Compute suggestion embeddings
            logging.info(f"\n[4/{total_steps}] Computing suggestion embeddings...")
            self._suggestion_embeddings_cache = self._compute_suggestion_embeddings()
            
            # Step 5 & 6: Compute steering embeddings if needed
            if self.return_triplets and self.steering_mode:
                logging.info(f"\n[5/{total_steps}] Computing steering embeddings...")
                self._compute_steering_embeddings()
                
                logging.info(f"\n[6/{total_steps}] Validating cluster assignments...")
                self._validate_cluster_assignments()
        else:
            logging.info(f"\n[3/{total_steps}] Skipping embeddings (return_embeddings=False)")
            logging.info(f"[4/{total_steps}] Skipping embeddings")

        # Save to disk if requested
        if save_to_cache and self.cache_dir:
            logging.info("\nSaving to cache...")
            self._save_cache()

        logging.info("\n" + "=" * 60)
        logging.info("✓ Build complete!")
        logging.info(f"  - Total samples: {len(self)}")
        logging.info(f"  - Unique suggestions: {len(self._unique_suggestions_cache)}")
        if self.steering_mode:
            logging.info(f"  - Steering mode: {self.steering_mode.value}")
        if self.cache_dir and save_to_cache:
            logging.info(f"  - Cache location: {self.cache_dir}")
        logging.info("=" * 60)

        return self

    def _parse_all_suggestions(self) -> List[List[str]]:
        """Parse and filter all suggestions"""
        all_parsed = []

        for idx in tqdm(range(len(self)), desc="Parsing"):
            raw_suggestions = self.df.iloc[idx][self.suggestions_col]
            suggestions = parse_suggestions_safe(raw_suggestions)
            filtered = filter_suggestions(
                suggestions, self.min_confidence, self.suggestion_types
            )
            all_parsed.append(filtered)

        return all_parsed

    def _extract_unique_suggestions(self) -> List[str]:
        """Extract all unique suggestion terms"""
        if self._parsed_suggestions_cache is None:
            self._parsed_suggestions_cache = self._parse_all_suggestions()

        unique_terms = set()
        for suggestions in self._parsed_suggestions_cache:
            unique_terms.update(suggestions)

        unique_list = sorted(list(unique_terms))
        logging.info(f"Found {len(unique_list)} unique suggestions")
        return unique_list

    def _compute_text_embeddings(self) -> List[Dict[str, np.ndarray]]:
        """Compute embeddings for all source texts and questions"""
        all_embeddings = []

        total_rows = len(self)
        for start_idx in tqdm(
            range(0, total_rows, self.chunksize), desc="Embedding texts"
        ):
            end_idx = min(start_idx + self.chunksize, total_rows)

            # Get batch
            batch_df = self.df.iloc[start_idx:end_idx]
            sources = batch_df[self.source_col].astype(str).tolist()
            questions = batch_df[self.question_col].astype(str).tolist()

            # Embed batch
            source_embeds = self.embedding_model.embed_documents(sources)
            question_embeds = self.embedding_model.embed_documents(questions)

            # Store
            for s_emb, q_emb in zip(source_embeds, question_embeds):
                all_embeddings.append(
                    {
                        "source": np.array(s_emb, dtype=np.float32),
                        "question": np.array(q_emb, dtype=np.float32),
                    }
                )

        return all_embeddings

    def _compute_suggestion_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute embeddings for all unique suggestions"""
        if self._unique_suggestions_cache is None:
            self._unique_suggestions_cache = self._extract_unique_suggestions()

        embeddings_dict = {}
        unique_suggestions = self._unique_suggestions_cache

        total_suggestions = len(unique_suggestions)
        for start_idx in tqdm(
            range(0, total_suggestions, self.chunksize), desc="Embedding suggestions"
        ):
            end_idx = min(start_idx + self.chunksize, total_suggestions)

            # Get batch
            batch = unique_suggestions[start_idx:end_idx]

            # Embed batch
            embeds = self.embedding_model.embed_documents(batch)

            # Store
            for term, emb in zip(batch, embeds):
                embeddings_dict[term] = np.array(emb, dtype=np.float32)

        return embeddings_dict

    def _compute_steering_embeddings(self):
        """Compute steering embeddings based on steering mode."""
        if not self.steering_mode or not self.return_embeddings:
            return

        if self.steering_mode == SteeringMode.ZERO:
            # No computation needed for zero mode
            return

        elif self.steering_mode == SteeringMode.CLUSTER_DESCRIPTOR:
            # Compute embeddings for cluster descriptors
            if not self.cluster_descriptors:
                logging.warning("No cluster descriptors provided for CLUSTER_DESCRIPTOR mode")
                return

            self._cluster_descriptor_embeddings_cache = {}
            
            all_descriptors = []
            descriptor_to_cluster = {}
            
            for cluster_id, descriptors in self.cluster_descriptors.items():
                if descriptors:
                    # Use first descriptor as representative
                    descriptor = descriptors[0]
                    all_descriptors.append(descriptor)
                    descriptor_to_cluster[descriptor] = cluster_id

            if all_descriptors:
                # Batch embed all descriptors
                for start_idx in tqdm(
                    range(0, len(all_descriptors), self.chunksize),
                    desc="Embedding cluster descriptors",
                ):
                    end_idx = min(start_idx + self.chunksize, len(all_descriptors))
                    batch = all_descriptors[start_idx:end_idx]
                    embeds = self.embedding_model.embed_documents(batch)

                    for descriptor, emb in zip(batch, embeds):
                        cluster_id = descriptor_to_cluster[descriptor]
                        self._cluster_descriptor_embeddings_cache[cluster_id] = np.array(
                            emb, dtype=np.float32
                        )

        elif self.steering_mode == SteeringMode.LLM_GENERATED:
            # Compute embeddings for LLM-generated texts
            if not self.llm_steering_texts:
                logging.warning("No LLM steering texts provided for LLM_GENERATED mode")
                return

            self._llm_steering_embeddings_cache = {}
            
            all_texts = list(self.llm_steering_texts.items())
            
            for start_idx in tqdm(
                range(0, len(all_texts), self.chunksize),
                desc="Embedding LLM steering texts",
            ):
                end_idx = min(start_idx + self.chunksize, len(all_texts))
                batch_items = all_texts[start_idx:end_idx]
                batch_texts = [text for _, text in batch_items]
                
                embeds = self.embedding_model.embed_documents(batch_texts)
                
                for (idx, _), emb in zip(batch_items, embeds):
                    self._llm_steering_embeddings_cache[idx] = np.array(
                        emb, dtype=np.float32
                    )

        elif self.steering_mode == SteeringMode.SUGGESTION:
            # Pre-compute steering embeddings for each sample (first suggestion)
            self._steering_embeddings_cache = []
            
            for idx in tqdm(range(len(self)), desc="Computing suggestion steering"):
                suggestions = self._parsed_suggestions_cache[idx]
                if suggestions and suggestions[0] in self._suggestion_embeddings_cache:
                    emb = self._suggestion_embeddings_cache[suggestions[0]]
                    self._steering_embeddings_cache.append(emb)
                else:
                    # Use zero vector if no suggestion available
                    embedding_dim = self._text_embeddings_cache[0]["question"].shape[0]
                    self._steering_embeddings_cache.append(
                        np.zeros(embedding_dim, dtype=np.float32)
                    )

        elif self.steering_mode == SteeringMode.MIXED:
            # For mixed mode, compute all component embeddings
            # Cluster descriptors
            if self.cluster_descriptors and self.steering_weights.get("cluster_descriptor", 0) > 0:
                logging.info("  Computing cluster descriptor embeddings for mixed mode...")
                self._cluster_descriptor_embeddings_cache = {}
                
                all_descriptors = []
                descriptor_to_cluster = {}
                
                for cluster_id, descriptors in self.cluster_descriptors.items():
                    if descriptors:
                        descriptor = descriptors[0]
                        all_descriptors.append(descriptor)
                        descriptor_to_cluster[descriptor] = cluster_id

                if all_descriptors:
                    for start_idx in range(0, len(all_descriptors), self.chunksize):
                        end_idx = min(start_idx + self.chunksize, len(all_descriptors))
                        batch = all_descriptors[start_idx:end_idx]
                        embeds = self.embedding_model.embed_documents(batch)

                        for descriptor, emb in zip(batch, embeds):
                            cluster_id = descriptor_to_cluster[descriptor]
                            self._cluster_descriptor_embeddings_cache[cluster_id] = np.array(
                                emb, dtype=np.float32
                            )

    def _validate_cluster_assignments(self):
        """Validate cluster assignments are consistent with data."""
        if not self.cluster_labels:
            return

        n_samples = len(self)
        n_assigned = len(self.cluster_labels)
        
        if n_assigned != n_samples:
            logging.warning(
                f"Cluster labels count ({n_assigned}) doesn't match dataset size ({n_samples})"
            )
        
        # Check for missing indices
        missing = set(range(n_samples)) - set(self.cluster_labels.keys())
        if missing:
            logging.warning(f"Missing cluster assignments for {len(missing)} samples")
        
        logging.info(f"Validated {n_assigned} cluster assignments")

    def _save_cache(self):
        """Save all processed data to cache"""
        if not self.cache_dir:
            raise ValueError("cache_dir must be set to save cache")

        # Save metadata
        metadata = {
            "version": self.version,
            "length": len(self),
            "source_col": self.source_col,
            "question_col": self.question_col,
            "suggestions_col": self.suggestions_col,
            "min_confidence": self.min_confidence,
            "suggestion_types": self.suggestion_types,
            "embedding_model_name": self.embedding_model_name,
            "num_unique_suggestions": (
                len(self._unique_suggestions_cache)
                if self._unique_suggestions_cache
                else 0
            ),
            "has_embeddings": self._text_embeddings_cache is not None,
            # Steering-related metadata
            "steering_mode": self.steering_mode.value if self.steering_mode else None,
            "return_triplets": self.return_triplets,
            "multi_label_mode": self.multi_label_mode,
            "has_cluster_labels": self.cluster_labels is not None,
            "has_cluster_descriptors": self.cluster_descriptors is not None,
            "has_llm_steering": self.llm_steering_texts is not None,
        }

        # Files to save
        cache_files = {
            "metadata.json": metadata,
            "parsed_suggestions.pkl": self._parsed_suggestions_cache,
            "unique_suggestions.pkl": self._unique_suggestions_cache,
        }

        if self._text_embeddings_cache is not None:
            cache_files["text_embeddings.pkl"] = self._text_embeddings_cache

        if self._suggestion_embeddings_cache is not None:
            cache_files["suggestion_embeddings.pkl"] = self._suggestion_embeddings_cache

        # Save steering-related caches
        if self._cluster_descriptor_embeddings_cache is not None:
            cache_files["cluster_descriptor_embeddings.pkl"] = (
                self._cluster_descriptor_embeddings_cache
            )

        if self._llm_steering_embeddings_cache is not None:
            cache_files["llm_steering_embeddings.pkl"] = (
                self._llm_steering_embeddings_cache
            )

        if self._steering_embeddings_cache is not None:
            cache_files["steering_embeddings.pkl"] = self._steering_embeddings_cache

        if self.cluster_labels is not None:
            cache_files["cluster_labels.pkl"] = self.cluster_labels

        if self.cluster_descriptors is not None:
            cache_files["cluster_descriptors.pkl"] = self.cluster_descriptors

        if self.llm_steering_texts is not None:
            cache_files["llm_steering_texts.pkl"] = self.llm_steering_texts

        # Save files
        for filename, data in cache_files.items():
            filepath = self.cache_dir / filename

            if filename.endswith(".json"):
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)
            else:
                with open(filepath, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logging.info(f"  ✓ Saved {filename}")

    def get_unique_suggestions(self) -> List[str]:
        """Get list of all unique suggestions"""
        if self._unique_suggestions_cache is None:
            self._unique_suggestions_cache = self._extract_unique_suggestions()
        return self._unique_suggestions_cache

    def get_suggestion_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all suggestion embeddings as a dictionary"""
        if self._suggestion_embeddings_cache is None:
            if not self.return_embeddings:
                raise ValueError("Dataset was created without embeddings")
            self._suggestion_embeddings_cache = self._compute_suggestion_embeddings()
        return self._suggestion_embeddings_cache


# ============================================================================
# DATASET CLASS - Load from Cache Only
# ============================================================================


class CachedDomainAssignDataset(Dataset):
    """
    PyTorch Dataset that loads pre-computed embeddings from cache.

    This class is lightweight and only loads from cache created by
    DomainAssignDatasetBuilder. No computation happens here.

    Supports both standard mode and cluster steering triplet mode.

    Usage:
        # After building cache with DomainAssignDatasetBuilder
        dataset = DomainAssignDataset(cache_dir='./cache')

        # Use with DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=32)

        for batch in loader:
            source = batch['source']
            question = batch['question']
            suggestions = batch['suggestions']
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        return_embeddings: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.return_embeddings = return_embeddings

        if not self.cache_dir.exists():
            raise ValueError(f"Cache directory does not exist: {self.cache_dir}")

        # Load metadata
        logging.info(f"Loading dataset from cache: {self.cache_dir}")
        self.metadata = self._load_metadata()
        self._length = self.metadata["length"]

        logging.info(f"  Version: {self.metadata['version']}")
        logging.info(f"  Samples: {self._length}")
        logging.info(f"  Unique suggestions: {self.metadata['num_unique_suggestions']}")

        # Load steering mode info
        steering_mode = self.metadata.get("steering_mode")
        self.steering_mode = SteeringMode(steering_mode) if steering_mode else None
        self.return_triplets = self.metadata.get("return_triplets", False)
        self.multi_label_mode = self.metadata.get("multi_label_mode", "hard")

        if self.steering_mode:
            logging.info(f"  Steering mode: {self.steering_mode.value}")

        # Load all cached data
        self.parsed_suggestions = self._load_pickle("parsed_suggestions.pkl")
        self.unique_suggestions = self._load_pickle("unique_suggestions.pkl")

        # Check if embeddings are available
        has_embeddings = self.metadata.get("has_embeddings", True)

        if return_embeddings:
            if not has_embeddings:
                raise ValueError(
                    "Cache was built without embeddings. "
                    "Set return_embeddings=False or rebuild cache with embeddings."
                )
            self.text_embeddings = self._load_pickle("text_embeddings.pkl")
            self.suggestion_embeddings = self._load_pickle("suggestion_embeddings.pkl")
        else:
            self.text_embeddings = None
            self.suggestion_embeddings = None

        # Load steering-related caches if available
        self.cluster_labels = self._load_pickle_optional("cluster_labels.pkl")
        self.cluster_descriptors = self._load_pickle_optional("cluster_descriptors.pkl")
        self.llm_steering_texts = self._load_pickle_optional("llm_steering_texts.pkl")
        self.cluster_descriptor_embeddings = self._load_pickle_optional(
            "cluster_descriptor_embeddings.pkl"
        )
        self.llm_steering_embeddings = self._load_pickle_optional(
            "llm_steering_embeddings.pkl"
        )
        self.steering_embeddings = self._load_pickle_optional("steering_embeddings.pkl")

        logging.info("✓ Dataset loaded successfully")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load and validate metadata"""
        metadata_path = self.cache_dir / "metadata.json"

        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            return json.load(f)

    def _load_pickle(self, filename: str) -> Any:
        """Load pickled data"""
        filepath = self.cache_dir / filename

        if not filepath.exists():
            raise ValueError(f"Cache file not found: {filepath}")

        with open(filepath, "rb") as f:
            return pickle.load(f)

    def _load_pickle_optional(self, filename: str) -> Optional[Any]:
        """Load pickled data if it exists, otherwise return None"""
        filepath = self.cache_dir / filename

        if not filepath.exists():
            return None

        with open(filepath, "rb") as f:
            return pickle.load(f)

    def get_unique_suggestions(self) -> List[str]:
        """Get list of all unique suggestions"""
        return self.unique_suggestions

    def get_suggestion_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all suggestion embeddings as a dictionary"""
        if self.suggestion_embeddings is None:
            raise ValueError("Dataset was loaded without embeddings")
        return self.suggestion_embeddings

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata"""
        return self.metadata

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            dict with keys (standard mode):
                - 'source': source embedding (if return_embeddings=True)
                - 'question': question embedding (if return_embeddings=True)
                - 'suggestions': list of suggestion embeddings (if return_embeddings=True)
                - 'suggestion_texts': list of suggestion terms (always included)
                - 'idx': index in dataset
            
            dict with keys (triplet mode):
                - 'base_embedding': question embedding
                - 'steering_embedding': steering embedding
                - 'target': cluster assignment
                - 'metadata': dict with steering info
                - 'idx': index in dataset
        """
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range [0, {self._length})")

        suggestions = self.parsed_suggestions[idx]

        if self.return_triplets and self.return_embeddings:
            # Triplet mode
            text_emb = self.text_embeddings[idx]
            
            # Get steering embedding
            steering_emb = None
            if self.steering_mode == SteeringMode.ZERO:
                steering_emb = np.zeros_like(text_emb["question"])
            elif self.steering_mode == SteeringMode.SUGGESTION:
                if self.steering_embeddings:
                    steering_emb = self.steering_embeddings[idx]
            elif self.steering_mode == SteeringMode.CLUSTER_DESCRIPTOR:
                if self.cluster_labels and idx in self.cluster_labels:
                    cluster_id = self._get_primary_cluster(idx)
                    if (
                        cluster_id is not None
                        and self.cluster_descriptor_embeddings
                        and cluster_id in self.cluster_descriptor_embeddings
                    ):
                        steering_emb = self.cluster_descriptor_embeddings[cluster_id]
            elif self.steering_mode == SteeringMode.LLM_GENERATED:
                if self.llm_steering_embeddings and idx in self.llm_steering_embeddings:
                    steering_emb = self.llm_steering_embeddings[idx]
            
            # Get target
            target = None
            if self.cluster_labels and idx in self.cluster_labels:
                assignment = self.cluster_labels[idx]
                if self.multi_label_mode == "hard":
                    if isinstance(assignment, int):
                        target = assignment
                    elif isinstance(assignment, list):
                        target = int(np.argmax(assignment))
                elif self.multi_label_mode == "soft":
                    if isinstance(assignment, list):
                        target = np.array(assignment, dtype=np.float32)
                    elif isinstance(assignment, int):
                        # This shouldn't happen but handle gracefully
                        target = assignment
            
            # Build metadata
            metadata = {
                "steering_mode": self.steering_mode.value if self.steering_mode else None,
                "suggestion_texts": suggestions,
                "sample_index": idx,
            }
            
            if self.cluster_labels and idx in self.cluster_labels:
                metadata["cluster_assignment"] = self.cluster_labels[idx]
                cluster_id = self._get_primary_cluster(idx)
                if (
                    cluster_id is not None
                    and self.cluster_descriptors
                    and cluster_id in self.cluster_descriptors
                ):
                    metadata["cluster_descriptors"] = self.cluster_descriptors[cluster_id]
            
            if self.llm_steering_texts and idx in self.llm_steering_texts:
                metadata["llm_steering_text"] = self.llm_steering_texts[idx]
            
            result = {
                "base_embedding": torch.from_numpy(text_emb["question"]),
                "source_embedding": torch.from_numpy(text_emb["source"]),
                "idx": idx,
                "metadata": metadata,
            }
            
            if steering_emb is not None:
                result["steering_embedding"] = torch.from_numpy(steering_emb)
            
            if target is not None:
                if isinstance(target, np.ndarray):
                    result["target"] = torch.from_numpy(target)
                else:
                    result["target"] = torch.tensor(target, dtype=torch.long)
            
            return result

        elif self.return_embeddings:
            # Standard embedding mode
            text_emb = self.text_embeddings[idx]

            suggestion_embeds = [
                self.suggestion_embeddings[term]
                for term in suggestions
                if term in self.suggestion_embeddings
            ]

            return {
                "source": torch.from_numpy(text_emb["source"]),
                "question": torch.from_numpy(text_emb["question"]),
                "suggestions": [torch.from_numpy(emb) for emb in suggestion_embeds],
                "suggestion_texts": suggestions,
                "idx": idx,
            }
        else:
            # Return just the suggestion terms
            return {"suggestion_texts": suggestions, "idx": idx}

    def _get_primary_cluster(self, idx: int) -> Optional[int]:
        """Get primary cluster assignment for a sample."""
        if not self.cluster_labels or idx not in self.cluster_labels:
            return None

        assignment = self.cluster_labels[idx]
        if isinstance(assignment, int):
            return assignment
        elif isinstance(assignment, list) and assignment:
            return int(np.argmax(assignment))
        return None


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def build_and_load_dataset(
    df: Union[pd.DataFrame, str, Path],
    embedding_model: Embeddings,
    cache_dir: Union[str, Path],
    force_rebuild: bool = False,
    **builder_kwargs,
) -> CachedDomainAssignDataset:
    """
    Convenience function to build cache if needed and load dataset.

    Args:
        df: DataFrame or path to CSV
        embedding_model: Embedding model to use
        cache_dir: Where to store/load cache
        force_rebuild: If True, rebuild even if cache exists
        **builder_kwargs: Additional arguments for DomainAssignDatasetBuilder

    Returns:
        DomainAssignDataset instance
    """
    cache_dir = Path(cache_dir)
    metadata_path = cache_dir / "metadata.json"

    # Check if cache exists and is valid
    should_build = force_rebuild or not metadata_path.exists()

    if not should_build:
        logging.info("Cache found, loading existing dataset...")
        try:
            return CachedDomainAssignDataset(cache_dir=cache_dir)
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}. Rebuilding...")

    logging.info("Building dataset cache...")
    dataset = BaseDomainAssignDataset(
        df=df, embedding_model=embedding_model, cache_dir=cache_dir, **builder_kwargs
    ).build()

    # Load the newly created cache
    return dataset


# TODO: Dataset for second stage
# class RAGSimpleClassificationDataset(Dataset):
#     logger = logging.getLogger(__name__)
#
#     def __init__(
#             self,
#             dataset: Optional[BaseRAGDatasetGenerator] = None,
#             embedding_model: Optional[Embeddings] = None,
#             sample_limit: Optional[int] = None,
#
#
#             # use_contrastive_samples: bool = True,
#             # use_positive_samples: bool = True,
#             # use_similar_samples: bool = False, # Or that as settings (dict, dataclass, kwargs)?
#     ):
#         self._data_df: Optional[pd.DataFrame] = None
#         self._dataset = dataset
#         self._embedding_model = embedding_model
#         self._sample_limit = sample_limit
#
#         self._validate_dataset()
#         # TODO:
#
#     def __len__(self):
#         return len(self._data_df)
#
#     def _validate_dataset(self):
#         pass #TODO
#
#     def _validate_csv_columns(self, df: pd.DataFrame, base_col: str, target_col: str, value_cols: Optional[List[str]]):
#         if base_col not in df.columns:
#             raise ValueError(f"Base column '{base_col}' not found in CSV.")
#         if target_col not in df.columns:
#             raise ValueError(f"Target column '{target_col}' not found in CSV.")
#         if value_cols:
#             for col in value_cols:
#                 if col not in df.columns:
#                     raise ValueError(f"Value column '{col}' not found in CSV.")
#
#     def samples_from_dataset(self):
#         pass # TODO: samples from BaseRAGDatasetGenerator (Chroma db)
#
#     def samples_from_csv(self,
#                          sample_csv_paths: Union[str, List[str]],
#                          base_col: str = 'question_text',
#                          target_col: str = 'source_text',
#                          value_cols: Optional[List[str]] = None,
#                          overwrite: bool = False
#                          ) -> pd.DataFrame:
#         if not self._embedding_model:
#             raise ValueError("Embedding model must be provided to load samples from CSV.")
#
#         if isinstance(sample_csv_paths, str):
#             sample_csv_paths = [sample_csv_paths]
#
#         # ✅ Keep these outside the file loop to cache across ALL files
#         base_embedd_dict = {}
#         target_embedd_dict = {}
#         samples_list = []  # ✅ Single list for all samples
#         total_samples = 0
#
#         for path in sample_csv_paths:
#             try:
#                 # ✅ For very large CSVs, consider chunked reading
#                 if self._is_large_csv(path):  # You'll need to implement this check
#                     df_chunks = pd.read_csv(path, chunksize=10000)
#                     df_iterator = df_chunks
#                 else:
#                     df_iterator = [pd.read_csv(path)]
#
#                 for df_chunk in df_iterator:
#                     self._validate_csv_columns(df_chunk, base_col, target_col, value_cols)
#
#                     # ✅ Process chunk efficiently
#                     chunk_samples = self._process_chunk(
#                         df_chunk, base_col, target_col, value_cols,
#                         base_embedd_dict, target_embedd_dict,
#                         total_samples
#                     )
#
#                     samples_list.extend(chunk_samples)
#                     total_samples += len(chunk_samples)
#
#                     # ✅ Early termination if limit reached
#                     if self._sample_limit and total_samples >= self._sample_limit:
#                         samples_list = samples_list[:self._sample_limit]
#                         break
#
#             except Exception as e:
#                 self.logger.error(f"Error processing file {path}: {e} (skipping dataset: {path})")
#
#             if self._sample_limit and total_samples >= self._sample_limit:
#                 break
#
#         # ✅ Create DataFrame only ONCE at the end
#         return pd.DataFrame(samples_list) if samples_list else pd.DataFrame()
#
#     def _process_chunk(self, df_chunk, base_col, target_col, value_cols,
#                        base_embedd_dict, target_embedd_dict, current_total):
#         """Process a chunk of data efficiently"""
#         chunk_samples = []
#
#         for row in tqdm(df_chunk.itertuples(), total=len(df_chunk),
#                         desc="Processing chunk", leave=False):
#
#             if self._sample_limit and (current_total + len(chunk_samples)) >= self._sample_limit:
#                 break
#
#             base_text = row[base_col]
#             target_text = row[target_col]
#             values = row[value_cols].tolist() if value_cols else []
#
#             # Get or compute embeddings
#             if base_text not in base_embedd_dict:
#                 base_embedd_dict[base_text] = self._embedding_model.embed_query(base_text)
#
#             if target_text not in target_embedd_dict:
#                 target_embedd_dict[target_text] = self._embedding_model.embed_query(target_text)
#
#             chunk_samples.append({
#                 'base_embedding': base_embedd_dict[base_text],
#                 'target_embedding': target_embedd_dict[target_text],
#                 'values': values,
#             })
#
#         return chunk_samples
#
#     def _is_large_csv(self, path, size_threshold_mb=100):
#         """Check if CSV file is large enough to warrant chunked processing"""
#         return os.path.getsize(path) > (size_threshold_mb * 1024 * 1024)
