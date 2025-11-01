import pandas as pd
from sklearn.metrics import roc_auc_score

from mlebench.competitions.utils import prepare_for_auroc_metric


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    roc_auc_inputs = prepare_for_auroc_metric(
        submission, answers, id_col="image_name", target_col="target"
    )
    return roc_auc_score(y_true=roc_auc_inputs["y_true"], y_score=roc_auc_inputs["y_score"])


def grade_lite(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    ROC AUC score with NaN handling for lite data.
    Score = raw_score * coverage (proportion of valid predictions)
    """
    import numpy as np
    
    roc_auc_inputs = prepare_for_auroc_metric(
        submission, answers, id_col="image_name", target_col="target"
    )
    
    y_true = roc_auc_inputs["y_true"]
    y_score = roc_auc_inputs["y_score"]
    
    # Check for NaN values in predictions
    nan_mask = np.isnan(y_score)
    total_samples = len(y_score)
    valid_samples = total_samples - nan_mask.sum()
    coverage = valid_samples / total_samples if total_samples > 0 else 0
    
    print(f"Melanoma detection ROC AUC evaluation summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Valid predictions: {valid_samples}")
    print(f"  NaN predictions: {nan_mask.sum()}")
    print(f"  Coverage: {coverage*100:.1f}%")
    
    if valid_samples == 0:
        print("No valid predictions - returning score of 0.0")
        return 0.0
    
    # Compute ROC AUC on valid samples only
    y_true_clean = y_true[~nan_mask]
    y_score_clean = y_score[~nan_mask]
    
    raw_score = roc_auc_score(y_true=y_true_clean, y_score=y_score_clean)
    final_score = raw_score * coverage
    
    print(f"  Raw ROC AUC: {raw_score:.4f}")
    print(f"  Final score (raw × coverage): {final_score:.4f}")
    
    return final_score