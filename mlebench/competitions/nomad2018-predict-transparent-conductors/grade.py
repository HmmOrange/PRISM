from typing import Tuple

import pandas as pd
from sklearn import metrics

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    assert (
        "formation_energy_ev_natom" in answers.columns
    ), "Expected 'formation_energy_ev_natom' column in answers"
    assert "bandgap_energy_ev" in answers.columns, "Expected 'bandgap_energy_ev' column in answers"
    if "formation_energy_ev_natom" not in submission.columns:
        raise InvalidSubmissionError("Expected 'formation_energy_ev_natom' column in submission")
    if "bandgap_energy_ev" not in submission.columns:
        raise InvalidSubmissionError("Expected 'bandgap_energy_ev' column in submission")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Expected {len(answers)} rows in submission, got {len(submission)}"
        )

    true_labels_formation, true_labels_bandgap = (
        answers["formation_energy_ev_natom"],
        answers["bandgap_energy_ev"],
    )
    predictions_formation, predictions_bandgap = (
        submission["formation_energy_ev_natom"],
        submission["bandgap_energy_ev"],
    )

    return true_labels_formation, true_labels_bandgap, predictions_formation, predictions_bandgap


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    (
        true_labels_formation,
        true_labels_bandgap,
        predictions_formation,
        predictions_bandgap,
    ) = prepare_for_metric(submission, answers)
    return (
        metrics.root_mean_squared_log_error(true_labels_formation, predictions_formation)
        + metrics.root_mean_squared_log_error(true_labels_bandgap, predictions_bandgap)
    ) / 2


def grade_lite(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Computes the average RMSLE for multi-target regression with NaN handling for lite data.
    
    Strategy: Evaluate on valid pairs and apply coverage-based adjustment
    - Handle NaN values in either target (formation_energy or bandgap_energy)
    - Compute RMSLE only on samples where both targets are valid
    - Apply coverage-based penalty to reflect missing data impact
    """
    import numpy as np
    
    (
        true_labels_formation,
        true_labels_bandgap,
        predictions_formation,
        predictions_bandgap,
    ) = prepare_for_metric(submission, answers)
    
    # Check for NaN values in predictions
    pred_formation_nan = pd.isna(predictions_formation)
    pred_bandgap_nan = pd.isna(predictions_bandgap)
    pred_nan_mask = pred_formation_nan | pred_bandgap_nan
    
    # Check for NaN values in ground truth
    true_formation_nan = pd.isna(true_labels_formation)
    true_bandgap_nan = pd.isna(true_labels_bandgap)
    true_nan_mask = true_formation_nan | true_bandgap_nan
    
    # Only keep samples where both predictions and ground truth are valid
    valid_mask = ~pred_nan_mask & ~true_nan_mask
    
    total_samples = len(true_labels_formation)
    valid_samples = valid_mask.sum()
    coverage = valid_samples / total_samples if total_samples > 0 else 0
    
    print(f"Multi-target regression evaluation summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Valid pairs: {valid_samples}")
    print(f"  Excluded (NaN): {total_samples - valid_samples}")
    print(f"  Coverage: {coverage*100:.1f}%")
    
    if pred_formation_nan.any():
        print(f"  NaN in formation_energy predictions: {pred_formation_nan.sum()}")
    if pred_bandgap_nan.any():
        print(f"  NaN in bandgap_energy predictions: {pred_bandgap_nan.sum()}")
    if true_formation_nan.any():
        print(f"  NaN in formation_energy ground truth: {true_formation_nan.sum()}")
    if true_bandgap_nan.any():
        print(f"  NaN in bandgap_energy ground truth: {true_bandgap_nan.sum()}")
    
    if valid_samples == 0:
        print("No valid pairs - returning RMSLE of 1.0 (high penalty)")
        return 1.0
    
    # Filter to valid samples only
    true_formation_clean = true_labels_formation[valid_mask]
    true_bandgap_clean = true_labels_bandgap[valid_mask]
    pred_formation_clean = predictions_formation[valid_mask]
    pred_bandgap_clean = predictions_bandgap[valid_mask]
    
    # Check for non-positive values (RMSLE requires positive values)
    formation_positive = (true_formation_clean > 0) & (pred_formation_clean > 0)
    bandgap_positive = (true_bandgap_clean > 0) & (pred_bandgap_clean > 0)
    
    if not formation_positive.all():
        print(f"Warning: Found {(~formation_positive).sum()} non-positive formation_energy values")
        # Clip to small positive value
        true_formation_clean = np.maximum(true_formation_clean, 1e-10)
        pred_formation_clean = np.maximum(pred_formation_clean, 1e-10)
    
    if not bandgap_positive.all():
        print(f"Warning: Found {(~bandgap_positive).sum()} non-positive bandgap_energy values")
        # Clip to small positive value  
        true_bandgap_clean = np.maximum(true_bandgap_clean, 1e-10)
        pred_bandgap_clean = np.maximum(pred_bandgap_clean, 1e-10)
    
    # Compute RMSLE for each target
    try:
        rmsle_formation = metrics.root_mean_squared_log_error(true_formation_clean, pred_formation_clean)
        rmsle_bandgap = metrics.root_mean_squared_log_error(true_bandgap_clean, pred_bandgap_clean)
        raw_score = (rmsle_formation + rmsle_bandgap) / 2
    except Exception as e:
        print(f"Error computing RMSLE: {e}")
        return 1.0  # High penalty for computation errors
    
    print(f"Target-wise RMSLE scores:")
    print(f"  Formation energy RMSLE: {rmsle_formation:.6f}")
    print(f"  Bandgap energy RMSLE: {rmsle_bandgap:.6f}")
    print(f"  Raw average RMSLE: {raw_score:.6f}")
    
    # Apply coverage-based adjustment
    # For RMSLE (lower is better), we apply penalty by increasing the score
    
    # Option 1: Linear penalty (increase RMSLE inversely with coverage)
    linear_penalty_score = raw_score / coverage if coverage > 0 else 1.0
    
    # Option 2: Conservative penalty (less harsh)
    conservative_penalty_score = raw_score / np.sqrt(coverage) if coverage > 0 else 1.0
    
    # Option 3: Threshold-based (no penalty if coverage > 90%)
    threshold_penalty_score = raw_score if coverage >= 0.9 else raw_score / coverage
    
    print(f"Coverage-adjusted scoring options:")
    print(f"  Raw score (on valid pairs): {raw_score:.6f}")
    print(f"  Linear penalty (÷coverage): {linear_penalty_score:.6f}")
    print(f"  Conservative penalty (÷√coverage): {conservative_penalty_score:.6f}")
    print(f"  Threshold penalty (90%): {threshold_penalty_score:.6f}")
    
    # Return conservative penalty as default (less harsh for high coverage)
    final_score = conservative_penalty_score
    print(f"Final RMSLE (conservative penalty): {final_score:.6f}")
    
    return final_score