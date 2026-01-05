"""Validation utilities for LLM-generated outputs in cluster steering and domain assessment.

This module provides validation functions to check:
- Schema compliance of JSON outputs
- Content quality and consistency
- Statistical properties of generated data
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

LOGGER = logging.getLogger(__name__)

# Optional numpy import for statistical functions
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    LOGGER.warning("numpy not available. Statistical functions will be limited.")


def validate_steering_text(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate steering text generation result.

    Parameters
    ----------
    result : Dict[str, Any]
        Steering text result dictionary

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_errors)
    """
    errors = []

    # Check required fields
    required_fields = [
        "cluster_id",
        "steering_text",
        "incorporated_descriptors",
        "confidence",
    ]
    for field in required_fields:
        if field not in result:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    # Validate types
    if not isinstance(result["cluster_id"], int):
        errors.append(f"cluster_id must be int, got {type(result['cluster_id'])}")

    if not isinstance(result["steering_text"], str):
        errors.append(
            f"steering_text must be str, got {type(result['steering_text'])}"
        )
    elif len(result["steering_text"].strip()) == 0:
        errors.append("steering_text is empty")

    if not isinstance(result["incorporated_descriptors"], list):
        errors.append(
            f"incorporated_descriptors must be list, got {type(result['incorporated_descriptors'])}"
        )

    if not isinstance(result["confidence"], (int, float)):
        errors.append(f"confidence must be numeric, got {type(result['confidence'])}")
    elif not 0.0 <= result["confidence"] <= 1.0:
        errors.append(
            f"confidence must be in [0, 1], got {result['confidence']}"
        )

    return len(errors) == 0, errors


def validate_question_rephrase(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate question rephrase result.

    Parameters
    ----------
    result : Dict[str, Any]
        Question rephrase result dictionary

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_errors)
    """
    errors = []

    # Check required fields
    required_fields = [
        "original_question",
        "target_cluster_id",
        "rephrased_question",
        "genre_shift",
        "preserved_intent",
        "confidence",
    ]
    for field in required_fields:
        if field not in result:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    # Validate types and content
    if not isinstance(result["original_question"], str):
        errors.append(
            f"original_question must be str, got {type(result['original_question'])}"
        )

    if not isinstance(result["target_cluster_id"], int):
        errors.append(
            f"target_cluster_id must be int, got {type(result['target_cluster_id'])}"
        )

    if not isinstance(result["rephrased_question"], str):
        errors.append(
            f"rephrased_question must be str, got {type(result['rephrased_question'])}"
        )
    elif len(result["rephrased_question"].strip()) == 0:
        errors.append("rephrased_question is empty")

    # Check that rephrased is different from original
    if (
        result.get("original_question", "").strip().lower()
        == result.get("rephrased_question", "").strip().lower()
    ):
        errors.append("rephrased_question is identical to original_question")

    if not isinstance(result["confidence"], (int, float)):
        errors.append(f"confidence must be numeric, got {type(result['confidence'])}")
    elif not 0.0 <= result["confidence"] <= 1.0:
        errors.append(
            f"confidence must be in [0, 1], got {result['confidence']}"
        )

    return len(errors) == 0, errors


def validate_ambiguity_resolution(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate ambiguity resolution result.

    Parameters
    ----------
    result : Dict[str, Any]
        Ambiguity resolution result dictionary

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_errors)
    """
    errors = []

    # Check required fields
    required_fields = [
        "question",
        "is_ambiguous",
        "primary_cluster",
        "secondary_clusters",
        "recommendation",
        "explanation",
    ]
    for field in required_fields:
        if field not in result:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    # Validate types
    if not isinstance(result["is_ambiguous"], bool):
        errors.append(
            f"is_ambiguous must be bool, got {type(result['is_ambiguous'])}"
        )

    if not isinstance(result["primary_cluster"], dict):
        errors.append(
            f"primary_cluster must be dict, got {type(result['primary_cluster'])}"
        )
    else:
        # Validate primary cluster structure
        pc = result["primary_cluster"]
        for field in ["cluster_id", "reason", "confidence"]:
            if field not in pc:
                errors.append(f"primary_cluster missing field: {field}")

    if not isinstance(result["secondary_clusters"], list):
        errors.append(
            f"secondary_clusters must be list, got {type(result['secondary_clusters'])}"
        )

    if result["recommendation"] not in ["single-domain", "multi-domain"]:
        errors.append(
            f"recommendation must be 'single-domain' or 'multi-domain', "
            f"got {result['recommendation']}"
        )

    return len(errors) == 0, errors


def validate_json_file(
    file_path: Path, validation_func: callable
) -> Tuple[int, int, List[str]]:
    """
    Validate a JSON file containing LLM outputs.

    Parameters
    ----------
    file_path : Path
        Path to JSON file
    validation_func : callable
        Validation function to use

    Returns
    -------
    Tuple[int, int, List[str]]
        (total_count, valid_count, all_errors)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        LOGGER.error(f"Failed to load JSON file {file_path}: {e}")
        return 0, 0, [f"Failed to load file: {e}"]

    # Handle both single result and list of results
    if isinstance(data, list):
        results = data
    else:
        results = [data]

    total_count = len(results)
    valid_count = 0
    all_errors = []

    for idx, result in enumerate(results):
        is_valid, errors = validation_func(result)
        if is_valid:
            valid_count += 1
        else:
            all_errors.extend([f"Item {idx}: {err}" for err in errors])

    LOGGER.info(
        f"Validated {file_path.name}: {valid_count}/{total_count} valid "
        f"({100 * valid_count / total_count:.1f}%)"
    )

    return total_count, valid_count, all_errors


def compute_confidence_statistics(
    results: List[Dict[str, Any]], confidence_key: str = "confidence"
) -> Dict[str, float]:
    """
    Compute statistics for confidence scores.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of result dictionaries
    confidence_key : str
        Key for confidence value

    Returns
    -------
    Dict[str, float]
        Statistics dictionary with mean, std, min, max, median
    """
    confidences = [
        r[confidence_key]
        for r in results
        if confidence_key in r and isinstance(r[confidence_key], (int, float))
    ]

    if not confidences:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }

    if HAS_NUMPY:
        import numpy as np
        return {
            "count": len(confidences),
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "median": float(np.median(confidences)),
        }
    else:
        # Fallback implementation without numpy
        mean_val = sum(confidences) / len(confidences)
        sorted_conf = sorted(confidences)
        n = len(sorted_conf)
        median_val = (
            sorted_conf[n // 2]
            if n % 2 == 1
            else (sorted_conf[n // 2 - 1] + sorted_conf[n // 2]) / 2
        )
        
        # Simple std calculation
        variance = sum((x - mean_val) ** 2 for x in confidences) / len(confidences)
        std_val = variance ** 0.5
        
        return {
            "count": len(confidences),
            "mean": mean_val,
            "std": std_val,
            "min": min(confidences),
            "max": max(confidences),
            "median": median_val,
        }


def analyze_descriptor_usage(
    steering_results: List[Dict[str, Any]],
    available_descriptors: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Analyze descriptor usage in steering text generation.

    Parameters
    ----------
    steering_results : List[Dict[str, Any]]
        List of steering text results
    available_descriptors : Optional[Set[str]]
        Set of available descriptors for validation

    Returns
    -------
    Dict[str, Any]
        Analysis results with usage counts and statistics
    """
    descriptor_counts = {}
    total_descriptors = 0
    results_with_descriptors = 0

    for result in steering_results:
        descriptors = result.get("incorporated_descriptors", [])
        if descriptors:
            results_with_descriptors += 1
            total_descriptors += len(descriptors)

            for desc in descriptors:
                descriptor_counts[desc] = descriptor_counts.get(desc, 0) + 1

    avg_descriptors = (
        total_descriptors / results_with_descriptors if results_with_descriptors > 0 else 0
    )

    analysis = {
        "total_results": len(steering_results),
        "results_with_descriptors": results_with_descriptors,
        "total_descriptors_used": total_descriptors,
        "unique_descriptors": len(descriptor_counts),
        "avg_descriptors_per_result": avg_descriptors,
        "most_common_descriptors": sorted(
            descriptor_counts.items(), key=lambda x: x[1], reverse=True
        )[:10],
    }

    # Check for invalid descriptors if available_descriptors provided
    if available_descriptors:
        invalid_descriptors = set(descriptor_counts.keys()) - available_descriptors
        analysis["invalid_descriptors"] = list(invalid_descriptors)
        analysis["invalid_count"] = len(invalid_descriptors)

    return analysis


def validate_batch_results(
    results_dir: Path,
    output_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Validate all LLM output JSON files in a directory.

    Parameters
    ----------
    results_dir : Path
        Directory containing JSON result files
    output_report_path : Optional[Path]
        Path to save validation report

    Returns
    -------
    Dict[str, Any]
        Validation report with statistics and errors
    """
    if not results_dir.exists():
        LOGGER.error(f"Results directory does not exist: {results_dir}")
        return {"error": "Directory not found"}

    report = {
        "results_dir": str(results_dir),
        "files_processed": 0,
        "total_items": 0,
        "valid_items": 0,
        "file_reports": [],
    }

    # Find all JSON files
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        LOGGER.warning(f"No JSON files found in {results_dir}")
        return report

    for json_file in json_files:
        LOGGER.info(f"Validating {json_file.name}...")

        # Determine validation function based on filename
        if "steering" in json_file.name.lower():
            validation_func = validate_steering_text
        elif "rephrase" in json_file.name.lower():
            validation_func = validate_question_rephrase
        elif "ambiguity" in json_file.name.lower():
            validation_func = validate_ambiguity_resolution
        else:
            LOGGER.warning(f"Unknown file type: {json_file.name}, skipping")
            continue

        total, valid, errors = validate_json_file(json_file, validation_func)

        file_report = {
            "filename": json_file.name,
            "total_items": total,
            "valid_items": valid,
            "validation_rate": valid / total if total > 0 else 0,
            "errors": errors[:20],  # Limit to first 20 errors
        }

        report["file_reports"].append(file_report)
        report["files_processed"] += 1
        report["total_items"] += total
        report["valid_items"] += valid

    # Compute overall statistics
    report["overall_validation_rate"] = (
        report["valid_items"] / report["total_items"]
        if report["total_items"] > 0
        else 0
    )

    # Save report if requested
    if output_report_path:
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        LOGGER.info(f"Validation report saved to {output_report_path}")

    return report
