import pandas as pd
from sklearn.metrics import accuracy_score

from mlebench.competitions.utils import prepare_for_accuracy_metric


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    accuracy_inputs = prepare_for_accuracy_metric(
        submission=submission, answers=answers, target_column="Cover_Type", id_column="Id"
    )
    return accuracy_score(**accuracy_inputs)


def grade_lite(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Accuracy score with NaN handling for lite data.
    Score = raw_score * coverage (proportion of valid predictions)
    """
    import numpy as np
    
    accuracy_inputs = prepare_for_accuracy_metric(
        submission=submission, answers=answers, target_column="Cover_Type", id_column="Id"
    )
    
    y_true = accuracy_inputs["y_true"]
    y_pred = accuracy_inputs["y_pred"]
    
    # Check for NaN values in predictions
    nan_mask = np.isnan(y_pred)
    total_samples = len(y_pred)
    valid_samples = total_samples - nan_mask.sum()
    coverage = valid_samples / total_samples if total_samples > 0 else 0
    
    print(f"Accuracy evaluation summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Valid predictions: {valid_samples}")
    print(f"  NaN predictions: {nan_mask.sum()}")
    print(f"  Coverage: {coverage*100:.1f}%")
    
    if valid_samples == 0:
        print("No valid predictions - returning score of 0.0")
        return 0.0
    
    # Compute accuracy on valid samples only
    y_true_clean = y_true[~nan_mask]
    y_pred_clean = y_pred[~nan_mask]
    
    raw_score = accuracy_score(y_true_clean, y_pred_clean)
    final_score = raw_score * coverage
    
    print(f"  Raw accuracy: {raw_score:.4f}")
    print(f"  Final score (raw × coverage): {final_score:.4f}")
    
    return final_score