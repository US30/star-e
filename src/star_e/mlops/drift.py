"""Data and model drift detection utilities."""

from typing import Literal

import numpy as np
from scipy import stats


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Calculate Population Stability Index (PSI).

    PSI measures how much a distribution has shifted between two samples.
    It's widely used in financial services for monitoring model inputs.

    PSI Interpretation:
    - < 0.1: No significant change
    - 0.1 - 0.25: Moderate shift, monitor closely
    - > 0.25: Significant shift, investigate and potentially retrain

    Formula:
        PSI = Σ (actual_% - expected_%) * ln(actual_% / expected_%)

    Args:
        expected: Reference distribution (e.g., training data)
        actual: Current distribution (e.g., production data)
        bins: Number of bins for discretization
        eps: Small value to avoid log(0)

    Returns:
        PSI value
    """
    expected = np.asarray(expected).flatten()
    actual = np.asarray(actual).flatten()

    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        raise ValueError("Empty arrays after removing NaN values")

    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Calculate percentages in each bin
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    # Avoid division by zero and log(0)
    expected_percents = np.clip(expected_percents, eps, 1 - eps)
    actual_percents = np.clip(actual_percents, eps, 1 - eps)

    # Calculate PSI
    psi = np.sum(
        (actual_percents - expected_percents)
        * np.log(actual_percents / expected_percents)
    )

    return float(psi)


def ks_test_drift(
    reference: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """
    Kolmogorov-Smirnov test for distribution drift.

    The KS test compares the empirical cumulative distribution functions
    of two samples and returns the maximum difference (D statistic).

    Args:
        reference: Reference distribution
        current: Current distribution
        threshold: P-value threshold for drift detection

    Returns:
        Dictionary with:
            - statistic: KS D statistic
            - pvalue: P-value
            - is_drift: Whether drift is detected
    """
    reference = np.asarray(reference).flatten()
    current = np.asarray(current).flatten()

    # Remove NaN
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    statistic, pvalue = stats.ks_2samp(reference, current)

    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "is_drift": pvalue < threshold,
    }


def wasserstein_distance(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    """
    Calculate Wasserstein distance (Earth Mover's Distance).

    Measures the minimum "work" needed to transform one distribution
    into another. More sensitive to small shifts than KS test.

    Args:
        reference: Reference distribution
        current: Current distribution

    Returns:
        Wasserstein distance
    """
    reference = np.asarray(reference).flatten()
    current = np.asarray(current).flatten()

    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    return float(stats.wasserstein_distance(reference, current))


def detect_drift(
    reference: np.ndarray,
    current: np.ndarray,
    method: Literal["psi", "ks", "wasserstein"] = "psi",
    psi_threshold: float = 0.1,
    ks_threshold: float = 0.05,
) -> dict:
    """
    Detect distribution drift using specified method.

    Args:
        reference: Reference distribution
        current: Current distribution
        method: Detection method ("psi", "ks", or "wasserstein")
        psi_threshold: PSI threshold for drift
        ks_threshold: KS test p-value threshold

    Returns:
        Dictionary with drift detection results
    """
    if method == "psi":
        psi = calculate_psi(reference, current)
        return {
            "method": "psi",
            "value": psi,
            "threshold": psi_threshold,
            "is_drift": psi > psi_threshold,
            "severity": "none" if psi < 0.1 else ("moderate" if psi < 0.25 else "severe"),
        }

    elif method == "ks":
        result = ks_test_drift(reference, current, ks_threshold)
        return {
            "method": "ks",
            "value": result["statistic"],
            "pvalue": result["pvalue"],
            "threshold": ks_threshold,
            "is_drift": result["is_drift"],
        }

    elif method == "wasserstein":
        distance = wasserstein_distance(reference, current)
        # Normalize by reference std for interpretability
        ref_std = np.std(reference)
        normalized = distance / ref_std if ref_std > 0 else distance

        return {
            "method": "wasserstein",
            "value": distance,
            "normalized": normalized,
            "is_drift": normalized > 0.5,  # Heuristic threshold
        }

    else:
        raise ValueError(f"Unknown method: {method}")


def monitor_features(
    reference_df: dict[str, np.ndarray],
    current_df: dict[str, np.ndarray],
    method: Literal["psi", "ks"] = "psi",
) -> dict[str, dict]:
    """
    Monitor multiple features for drift.

    Args:
        reference_df: Dict mapping feature names to reference arrays
        current_df: Dict mapping feature names to current arrays
        method: Detection method

    Returns:
        Dict mapping feature names to drift results
    """
    results = {}

    for feature in reference_df:
        if feature not in current_df:
            continue

        results[feature] = detect_drift(
            reference_df[feature],
            current_df[feature],
            method=method,
        )

    # Summary statistics
    n_drifted = sum(1 for r in results.values() if r.get("is_drift", False))

    return {
        "features": results,
        "n_features": len(results),
        "n_drifted": n_drifted,
        "drift_rate": n_drifted / len(results) if results else 0,
    }


def concept_drift_detection(
    predictions: np.ndarray,
    actuals: np.ndarray,
    window_size: int = 50,
    threshold: float = 2.0,
) -> dict:
    """
    Detect concept drift by monitoring prediction errors over time.

    Uses a rolling window to detect when prediction errors significantly
    increase compared to historical performance.

    Args:
        predictions: Model predictions
        actuals: Actual values
        window_size: Rolling window size
        threshold: Number of standard deviations for drift detection

    Returns:
        Dictionary with drift detection results
    """
    errors = np.abs(predictions - actuals)

    if len(errors) < window_size * 2:
        return {
            "is_drift": False,
            "message": "Insufficient data for drift detection",
        }

    # Calculate rolling mean error
    reference_error = np.mean(errors[:window_size])
    reference_std = np.std(errors[:window_size])

    current_error = np.mean(errors[-window_size:])

    z_score = (current_error - reference_error) / reference_std if reference_std > 0 else 0

    return {
        "reference_error": float(reference_error),
        "current_error": float(current_error),
        "z_score": float(z_score),
        "threshold": threshold,
        "is_drift": abs(z_score) > threshold,
        "direction": "increase" if z_score > 0 else "decrease",
    }
