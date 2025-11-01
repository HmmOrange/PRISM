from pandas import DataFrame
from sklearn.metrics import root_mean_squared_error

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(submission: DataFrame, answers: DataFrame) -> dict:

    assert "fare_amount" in answers.columns, "Answers should have a fare_amount column"
    assert "key" in answers.columns, "Answers should have a key column"
    if "fare_amount" not in submission.columns:
        raise InvalidSubmissionError("Submission should have a fare_amount column")
    if "key" not in submission.columns:
        raise InvalidSubmissionError("Submission should have a key column")

    # Sort by 'key' to ensure alignment
    submission = submission.sort_values("key")
    answers = answers.sort_values("key")

    y_true = answers["fare_amount"]
    y_pred = submission["fare_amount"]

    return y_true, y_pred


def grade(submission: DataFrame, answers: DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    return root_mean_squared_error(y_true, y_pred)


def grade_lite(submission: DataFrame, answers: DataFrame) -> float:
    """
    RMSE score with NaN handling for lite data.
    Score = raw_score * coverage (proportion of valid predictions)
    """
    import numpy as np
    
    y_true, y_pred = prepare_for_metric(submission, answers)
    
    # Check for NaN values in predictions
    nan_mask = np.isnan(y_pred)
    total_samples = len(y_pred)
    valid_samples = total_samples - nan_mask.sum()
    coverage = valid_samples / total_samples if total_samples > 0 else 0
    
    print(f"RMSE evaluation summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Valid predictions: {valid_samples}")
    print(f"  NaN predictions: {nan_mask.sum()}")
    print(f"  Coverage: {coverage*100:.1f}%")
    
    if valid_samples == 0:
        print("No valid predictions - returning high RMSE penalty")
        return 1000.0  # High penalty for no predictions
    
    # Compute RMSE on valid samples only
    y_true_clean = y_true[~nan_mask]
    y_pred_clean = y_pred[~nan_mask]
    
    raw_score = root_mean_squared_error(y_true_clean, y_pred_clean)
    # For RMSE (lower is better), we want to penalize missing predictions
    # Add penalty proportional to missing data
    missing_penalty = raw_score * (1 - coverage)
    final_score = raw_score + missing_penalty
    
    print(f"  Raw RMSE: {raw_score:.4f}")
    print(f"  Missing penalty: {missing_penalty:.4f}")
    print(f"  Final score (raw + penalty): {final_score:.4f}")
    
    return final_score