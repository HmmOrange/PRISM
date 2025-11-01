from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from mlebench.grade_helpers import InvalidSubmissionError


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    # answers checks
    assert "id" in answers.columns, f"Answers is missing `id` column"
    assert "label" in answers.columns, f"Answers is missing `label` column"

    # submission checks
    if "id" not in submission.columns:
        raise InvalidSubmissionError(f"Submission is missing `id` column")
    if "label" not in submission.columns:
        raise InvalidSubmissionError(f"Submission is missing `label` column")
    if set(submission["id"]) != set(answers["id"]):
        raise InvalidSubmissionError("Submission and answers have different id's")
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission and answers have different lengths")
    if not ((submission["label"] >= 0) & (submission["label"] <= 1)).all():
        raise InvalidSubmissionError(
            "All values in submission `label` column must be between 0 and 1."
        )

    # sort by id to ensure correct order
    submission = submission.sort_values("id")
    answers = answers.sort_values("id")

    y_true = answers["label"]
    y_pred = submission["label"]
    return y_true, y_pred


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    y_true, y_pred = prepare_for_metric(submission, answers)
    score = log_loss(y_true=y_true, y_pred=y_pred)
    return score


def grade_lite(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Log Loss score with NaN handling for lite data.
    Score = raw_score * coverage (proportion of valid predictions)
    """
    import numpy as np
    
    y_true, y_pred = prepare_for_metric(submission, answers)
    
    # Check for NaN values in predictions
    nan_mask = np.isnan(y_pred)
    total_samples = len(y_pred)
    valid_samples = total_samples - nan_mask.sum()
    coverage = valid_samples / total_samples if total_samples > 0 else 0
    
    print(f"Log Loss evaluation summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Valid predictions: {valid_samples}")
    print(f"  NaN predictions: {nan_mask.sum()}")
    print(f"  Coverage: {coverage*100:.1f}%")
    
    if valid_samples == 0:
        print("No valid predictions - returning high log loss penalty")
        return 10.0  # High penalty for no predictions
    
    # Compute log loss on valid samples only
    y_true_clean = y_true[~nan_mask]
    y_pred_clean = y_pred[~nan_mask]
    
    raw_score = log_loss(y_true=y_true_clean, y_pred=y_pred_clean)
    # For log loss (lower is better), we want to penalize missing predictions
    # Add penalty proportional to missing data
    missing_penalty = raw_score * (1 - coverage)
    final_score = raw_score + missing_penalty
    
    print(f"  Raw log loss: {raw_score:.4f}")
    print(f"  Missing penalty: {missing_penalty:.4f}")
    print(f"  Final score (raw + penalty): {final_score:.4f}")
    
    return final_score