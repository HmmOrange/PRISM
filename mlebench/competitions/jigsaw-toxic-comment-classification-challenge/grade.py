import pandas as pd
from sklearn.metrics import roc_auc_score

from mlebench.grade_helpers import InvalidSubmissionError

from .classes import CLASSES


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    classes_set = set(CLASSES)
    id_col = "id"

    # submission checks
    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission is missing the column: {id_col}")
    sub_cols_set = set(submission.columns)
    if not classes_set.issubset(sub_cols_set):
        raise InvalidSubmissionError(
            f"Submission is missing the following columns: {classes_set - sub_cols_set}"
        )

    # answers checks
    assert id_col in answers.columns, f"Answers is missing `{id_col}` column"
    ans_cols_set = set(answers.columns)
    assert classes_set.issubset(
        ans_cols_set
    ), f"Answers is missing the following columns: {classes_set - ans_cols_set}"
    assert len(submission) == len(answers), "Submission and answers have different lengths"

    submission = submission.set_index("id").sort_index()
    answers = answers.set_index("id").sort_index()

    # skip rows marked with -1 in y_true
    # when it happens entire row is marked so we can check negative sum of the row
    keep_mask = answers[CLASSES].sum(axis=1) >= 0
    answers = answers[keep_mask]
    submission = submission[keep_mask]

    roc_auc_inputs = {
        "y_score": submission.to_numpy(),
        "y_true": answers.to_numpy(),
        # metric for each column, then average across columns
        "average": "macro",
    }

    return roc_auc_inputs


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Computes the column-wise mean ROC AUC score for the submission.
    """
    roc_auc_inputs = prepare_for_metric(submission, answers)
    return roc_auc_score(**roc_auc_inputs)

def grade_lite(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Computes the column-wise mean ROC AUC score for the submission.
    
    For columns where ROC AUC cannot be computed (only 1 class present),
    those columns are excluded from the macro average calculation.
    """
    import numpy as np
    
    roc_auc_inputs = prepare_for_metric(submission, answers)
    
    # Extract data
    y_true = roc_auc_inputs["y_true"]
    y_score = roc_auc_inputs["y_score"]
    
    # Compute ROC AUC for each column individually, handling NaN cases
    column_scores = []
    for i, class_name in enumerate(CLASSES):
        y_true_col = y_true[:, i]
        y_score_col = y_score[:, i]
        
        # Check if this column has more than 1 unique class
        unique_classes = np.unique(y_true_col)
        if len(unique_classes) <= 1:
            # Skip columns with only 1 class (ROC AUC undefined)
            print(f"Warning: Column '{class_name}' has only {len(unique_classes)} unique class(es), skipping from ROC AUC calculation")
            continue
        
        try:
            score = roc_auc_score(y_true_col, y_score_col)
            column_scores.append(score)
            print(f"Column '{class_name}': ROC AUC = {score:.4f}")
        except Exception as e:
            print(f"Warning: Could not compute ROC AUC for column '{class_name}': {e}")
            continue
    
    if len(column_scores) == 0:
        raise ValueError("Cannot compute ROC AUC: no columns have sufficient class diversity")
    
    # Return macro average of valid column scores
    macro_avg = np.mean(column_scores)
    print(f"ROC AUC computed for {len(column_scores)}/{len(CLASSES)} toxicity classes, macro average: {macro_avg:.4f}")
    
    return macro_avg