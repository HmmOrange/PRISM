from pandas import DataFrame
from sklearn.metrics import cohen_kappa_score

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_grading(submission: DataFrame, answers: DataFrame, target_column: str) -> tuple:

    if target_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must contain the target column '{target_column}'")
    assert (
        target_column in answers.columns
    ), f"Answers must contain the target column '{target_column}'"
    if submission.empty:
        raise InvalidSubmissionError("Submission DataFrame should not be empty")
    assert not answers.empty, "Answers DataFrame should not be empty"
    if set(submission["id_code"]) != set(answers["id_code"]):
        raise InvalidSubmissionError("Submission must have the same id_codes as answers")

    # Ensure the submissions and answers are sorted by id_code
    submission_sorted = submission.sort_values(by="id_code")
    answers_sorted = answers.sort_values(by="id_code")

    # Extract the target columns
    y_pred = submission_sorted[target_column].values
    y_true = answers_sorted[target_column].values

    return y_pred, y_true


def grade(submission: DataFrame, answers: DataFrame) -> float:
    y_pred, y_true = prepare_for_grading(
        submission=submission, answers=answers, target_column="diagnosis"
    )
    return cohen_kappa_score(y_pred, y_true, weights="quadratic")


def grade_lite(submission: DataFrame, answers: DataFrame) -> float:
    """
    Computes the quadratic weighted Cohen's kappa score with NaN handling for lite data.
    
    Strategy: Evaluate on valid pairs and apply coverage-based adjustment
    - Compute score on valid prediction-ground truth pairs
    - Apply penalty based on coverage to reflect missing data impact
    - Provides multiple scoring options for different use cases
    """
    import numpy as np
    
    y_pred, y_true = prepare_for_grading(
        submission=submission, answers=answers, target_column="diagnosis"
    )
    
    # Check for NaN values
    pred_nan_mask = np.isnan(y_pred)
    true_nan_mask = np.isnan(y_true)
    
    # Only keep samples where both prediction and ground truth are valid
    valid_mask = ~pred_nan_mask & ~true_nan_mask
    
    # Filter to only valid pairs
    y_pred_clean = y_pred[valid_mask]
    y_true_clean = y_true[valid_mask]
    
    total_samples = len(y_pred)
    valid_samples = len(y_pred_clean)
    coverage = valid_samples / total_samples if total_samples > 0 else 0
    
    print(f"Evaluation summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Valid pairs: {valid_samples}")
    print(f"  Excluded (NaN): {total_samples - valid_samples}")
    print(f"  Coverage: {coverage*100:.1f}%")
    
    if valid_samples == 0:
        print("No valid pairs - returning score of 0.0")
        return 0.0
    
    # Ensure integer values for diagnosis levels
    y_pred_clean = np.round(y_pred_clean).astype(int)
    y_true_clean = np.round(y_true_clean).astype(int)
    
    # Clip values to valid diagnosis range [0, 4]
    y_pred_clean = np.clip(y_pred_clean, 0, 4)
    y_true_clean = np.clip(y_true_clean, 0, 4)
    
    # Check class diversity
    unique_pred = np.unique(y_pred_clean)
    unique_true = np.unique(y_true_clean)
    
    print(f"  Prediction classes: {unique_pred}")
    print(f"  Ground truth classes: {unique_true}")
    
    if len(unique_true) <= 1:
        print("Warning: Ground truth has only 1 unique class")
        if len(unique_pred) <= 1 and unique_pred[0] == unique_true[0]:
            raw_score = 1.0  # Perfect agreement
        else:
            raw_score = 0.0  # No agreement
    else:
        # Compute Cohen's kappa on valid pairs
        raw_score = cohen_kappa_score(y_pred_clean, y_true_clean, weights="quadratic")
    
    print(f"Raw kappa score: {raw_score:.4f}")
    
    # Apply coverage-based adjustment
    # Option 1: Linear penalty for missing data
    coverage_adjusted_score = raw_score * coverage
    
    # Option 2: Conservative penalty (square root to be less harsh)
    conservative_score = raw_score * np.sqrt(coverage)
    
    # Option 3: Threshold-based (no penalty if coverage > 90%)
    threshold_score = raw_score if coverage >= 0.9 else raw_score * coverage
    
    print(f"Scoring options:")
    print(f"  Raw score (on valid pairs): {raw_score:.4f}")
    print(f"  Simple (raw × coverage): {coverage_adjusted_score:.4f}")
    print(f"  Conservative (raw × √coverage): {conservative_score:.4f}")
    print(f"  Threshold (90% cutoff): {threshold_score:.4f}")
    
    # Return the conservative score as default (medical-appropriate)
    final_score = conservative_score
    print(f"Final score (conservative - medical appropriate): {final_score:.4f}")
    
    return final_score