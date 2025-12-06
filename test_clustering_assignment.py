"""
Integration test for Phase 1 & 2: Clustering Foundation and Source Assignment.

This script demonstrates and validates the complete workflow:
1. Create mock suggestion embeddings
2. Cluster suggestions (Phase 1)
3. Assign sources to clusters (Phase 2)
"""

import json
import tempfile
from pathlib import Path

import numpy as np


def test_phase_1_clustering():
    """Test Phase 1: Clustering Foundation."""
    print("=" * 60)
    print("Testing Phase 1: Clustering Foundation")
    print("=" * 60)
    
    from RAG_supporters.clustering import (
        SuggestionClusterer,
        cluster_suggestions_from_embeddings,
    )
    
    # Create mock suggestion embeddings
    np.random.seed(42)
    n_suggestions = 50
    embedding_dim = 384
    
    suggestions = [f"suggestion_{i}" for i in range(n_suggestions)]
    
    # Create clustered embeddings (3 clear groups)
    suggestion_embeddings = {}
    for i, sugg in enumerate(suggestions):
        if i < 20:
            # Cluster 0 around (1, 0, 0, ...)
            base = np.zeros(embedding_dim)
            base[0] = 1.0
            embedding = base + np.random.normal(0, 0.1, embedding_dim)
        elif i < 35:
            # Cluster 1 around (0, 1, 0, ...)
            base = np.zeros(embedding_dim)
            base[1] = 1.0
            embedding = base + np.random.normal(0, 0.1, embedding_dim)
        else:
            # Cluster 2 around (0, 0, 1, ...)
            base = np.zeros(embedding_dim)
            base[2] = 1.0
            embedding = base + np.random.normal(0, 0.1, embedding_dim)
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        suggestion_embeddings[sugg] = embedding
    
    print(f"\n✓ Created {len(suggestion_embeddings)} mock suggestion embeddings")
    
    # Test convenience function
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "clusters.json"
        
        clusterer, topics = cluster_suggestions_from_embeddings(
            suggestion_embeddings,
            n_clusters=3,
            algorithm="kmeans",
            n_descriptors=5,
            output_path=str(output_path),
            random_state=42,
        )
        
        print(f"\n✓ Clustered suggestions into {len(topics)} topics")
        
        # Verify topics were extracted
        for topic_id, descriptors in topics.items():
            print(f"  Topic {topic_id}: {len(descriptors)} descriptors")
            assert len(descriptors) == 5, "Should have 5 descriptors per topic"
        
        # Verify file was saved
        assert output_path.exists(), "Clustering results should be saved"
        print(f"✓ Results saved to {output_path}")
        
        # Load and verify
        with open(output_path, "r") as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert "cluster_assignments" in data
        assert "clusters" in data
        assert "cluster_stats" in data
        assert data["metadata"]["n_clusters"] == 3
        
        print("✓ Saved data structure is correct")
        
        # Test loading from results
        loaded_clusterer = SuggestionClusterer.from_results(str(output_path))
        assert len(loaded_clusterer.topics) == 3
        print("✓ Successfully loaded clusterer from saved results")
    
    # Test SuggestionClusterer class directly
    clusterer2 = SuggestionClusterer(
        algorithm="bisecting_kmeans",
        n_clusters=3,
        random_state=42,
    )
    clusterer2.fit(suggestion_embeddings)
    
    topics2 = clusterer2.extract_topic_descriptors(n_descriptors=10, metric="cosine")
    assert len(topics2) == 3
    for descriptors in topics2.values():
        assert len(descriptors) == 10
    
    print("✓ SuggestionClusterer class works correctly")
    
    # Get assignments
    assignments = clusterer2.get_cluster_assignments()
    assert len(assignments) == n_suggestions
    print(f"✓ Retrieved {len(assignments)} cluster assignments")
    
    # Get clusters
    clusters = clusterer2.get_clusters()
    assert len(clusters) == 3
    total_suggestions = sum(len(suggs) for suggs in clusters.values())
    assert total_suggestions == n_suggestions
    print(f"✓ Retrieved {len(clusters)} clusters")
    
    print("\n" + "=" * 60)
    print("Phase 1 Tests PASSED ✓")
    print("=" * 60 + "\n")
    
    return clusterer2


def test_phase_2_assignment(clusterer):
    """Test Phase 2: Source Assignment."""
    print("=" * 60)
    print("Testing Phase 2: Source Assignment")
    print("=" * 60)
    
    from RAG_supporters.clustering import (
        SourceAssigner,
        assign_sources_to_clusters,
    )
    
    # Get centroids from Phase 1
    centroids = clusterer.clusterer.get_centroids()
    print(f"\n✓ Retrieved {len(centroids)} cluster centroids")
    
    # Create mock source embeddings
    np.random.seed(123)
    n_sources = 10
    embedding_dim = centroids.shape[1]
    
    source_embeddings = {}
    for i in range(n_sources):
        source_id = f"source_{i}"
        # Create embeddings similar to clusters
        cluster_idx = i % 3
        embedding = centroids[cluster_idx] + np.random.normal(0, 0.05, embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        source_embeddings[source_id] = embedding
    
    print(f"✓ Created {len(source_embeddings)} mock source embeddings")
    
    # Test hard assignment
    print("\n--- Testing Hard Assignment ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "hard_assignments.json"
        
        assignments_hard = assign_sources_to_clusters(
            source_embeddings,
            centroids,
            assignment_mode="hard",
            threshold=0.3,
            output_path=str(output_path),
        )
        
        print(f"✓ Assigned {len(assignments_hard)} sources (hard mode)")
        
        # Verify hard assignments
        for source_id, assignment in assignments_hard.items():
            assert assignment["mode"] == "hard"
            clusters = assignment["clusters"]
            assert len(clusters) <= 1, "Hard assignment should assign to at most 1 cluster"
            if clusters:
                print(f"  {source_id} -> Cluster {clusters[0]}")
        
        # Verify file was saved
        assert output_path.exists()
        with open(output_path, "r") as f:
            data = json.load(f)
        assert "metadata" in data
        assert "statistics" in data
        assert "assignments" in data
        print("✓ Hard assignment results saved correctly")
    
    # Test soft assignment
    print("\n--- Testing Soft Assignment ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "soft_assignments.json"
        
        assignments_soft = assign_sources_to_clusters(
            source_embeddings,
            centroids,
            assignment_mode="soft",
            threshold=0.1,
            temperature=1.0,
            metric="cosine",
            output_path=str(output_path),
            metadata={"test": "phase_2"},
        )
        
        print(f"✓ Assigned {len(assignments_soft)} sources (soft mode)")
        
        # Verify soft assignments
        for source_id, assignment in assignments_soft.items():
            assert assignment["mode"] == "soft"
            assert "probabilities" in assignment
            assert "primary_cluster" in assignment
            
            # Verify probabilities sum to ~1.0
            probs = list(assignment["probabilities"].values())
            assert abs(sum(probs) - 1.0) < 0.01, "Probabilities should sum to 1"
            
            # Check multi-cluster membership
            clusters = assignment["clusters"]
            if len(clusters) > 1:
                print(f"  {source_id} -> Clusters {clusters} (multi-subspace)")
            else:
                print(f"  {source_id} -> Cluster {clusters}")
        
        # Verify file and metadata
        assert output_path.exists()
        with open(output_path, "r") as f:
            data = json.load(f)
        assert data["metadata"]["test"] == "phase_2"
        print("✓ Soft assignment results saved with metadata")
    
    # Test SourceAssigner class directly
    print("\n--- Testing SourceAssigner Class ---")
    assigner = SourceAssigner(
        cluster_centroids=centroids,
        assignment_mode="soft",
        threshold=0.15,
        temperature=0.8,
        metric="cosine",
    )
    
    # Assign single source
    source_emb = source_embeddings["source_0"]
    single_assignment = assigner.assign_source(source_emb, return_probabilities=True)
    assert "primary_cluster" in single_assignment
    assert "clusters" in single_assignment
    assert "probabilities" in single_assignment
    print(f"✓ Single source assignment: cluster {single_assignment['primary_cluster']}")
    
    # Batch assignment
    batch_assignments = assigner.assign_sources_batch(source_embeddings)
    assert len(batch_assignments) == n_sources
    print(f"✓ Batch assigned {len(batch_assignments)} sources")
    
    # Test with different temperature
    print("\n--- Testing Temperature Effects ---")
    assigner_hot = SourceAssigner(
        cluster_centroids=centroids,
        assignment_mode="soft",
        temperature=2.0,  # High temperature = more uniform
        metric="cosine",
    )
    
    assigner_cold = SourceAssigner(
        cluster_centroids=centroids,
        assignment_mode="soft",
        temperature=0.5,  # Low temperature = more peaked
        metric="cosine",
    )
    
    assignment_hot = assigner_hot.assign_source(source_emb)
    assignment_cold = assigner_cold.assign_source(source_emb)
    
    # Hot should be more uniform (lower max probability)
    max_prob_hot = max(assignment_hot["probabilities"].values())
    max_prob_cold = max(assignment_cold["probabilities"].values())
    
    print(f"  High temp (T=2.0): max prob = {max_prob_hot:.3f}")
    print(f"  Low temp (T=0.5): max prob = {max_prob_cold:.3f}")
    assert max_prob_cold > max_prob_hot, "Cold should be more peaked"
    print("✓ Temperature scaling works correctly")
    
    print("\n" + "=" * 60)
    print("Phase 2 Tests PASSED ✓")
    print("=" * 60 + "\n")


def test_complete_workflow():
    """Test complete Phase 1 & 2 workflow."""
    print("\n" + "=" * 60)
    print("Testing Complete Workflow")
    print("=" * 60 + "\n")
    
    # Phase 1
    clusterer = test_phase_1_clustering()
    
    # Phase 2
    test_phase_2_assignment(clusterer)
    
    print("=" * 60)
    print("ALL TESTS PASSED ✓✓✓")
    print("=" * 60)
    print("\nPhase 1 & 2 implementation is working correctly!")
    print("- Clustering foundation ✓")
    print("- Topic descriptor extraction ✓")
    print("- Source assignment (hard & soft) ✓")
    print("- Multi-subspace membership ✓")
    print("- Result persistence ✓")


if __name__ == "__main__":
    test_complete_workflow()
