import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mlebench.competitions.utils import prepare_for_auroc_metric
from mlebench.grade_helpers import InvalidSubmissionError

from .classes import CLASSES


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    for class_name in CLASSES:
        assert class_name in answers.columns, f"Missing class {class_name} in answers."
        if class_name not in submission.columns:
            raise InvalidSubmissionError(f"Class {class_name} is not in the submission.")
    assert len(submission) == len(
        answers
    ), f"Expected {len(answers)} rows in submission, got {len(submission)}."

    roc_auc_scores = []
    for class_name in CLASSES:
        roc_auc_inputs = prepare_for_auroc_metric(
            submission=submission,
            answers=answers,
            id_col="StudyInstanceUID",
            target_col=class_name,
        )
        roc_auc_scores.append(roc_auc_score(roc_auc_inputs["y_true"], roc_auc_inputs["y_score"]))

    score = np.average(roc_auc_scores)
    return score


def grade_lite(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Computes the column-wise mean ROC AUC score for the submission with lite data handling.
    
    For catheter line classes where ROC AUC cannot be computed (only 1 class present),
    those classes are excluded from the macro average calculation.
    This is specifically designed for lite datasets that may have insufficient class diversity.
    """
    for class_name in CLASSES:
        assert class_name in answers.columns, f"Missing class {class_name} in answers."
        if class_name not in submission.columns:
            raise InvalidSubmissionError(f"Class {class_name} is not in the submission.")
    assert len(submission) == len(
        answers
    ), f"Expected {len(answers)} rows in submission, got {len(submission)}."

    roc_auc_scores = []
    valid_classes = []
    
    for class_name in CLASSES:
        roc_auc_inputs = prepare_for_auroc_metric(
            submission=submission,
            answers=answers,
            id_col="StudyInstanceUID",
            target_col=class_name,
        )
        
        # Check if this class has more than 1 unique value
        unique_classes = np.unique(roc_auc_inputs["y_true"])
        if len(unique_classes) <= 1:
            print(f"Warning: Class '{class_name}' has only {len(unique_classes)} unique value(s), skipping from ROC AUC calculation")
            continue
        
        try:
            score = roc_auc_score(roc_auc_inputs["y_true"], roc_auc_inputs["y_score"])
            roc_auc_scores.append(score)
            valid_classes.append(class_name)
            print(f"Class '{class_name}': ROC AUC = {score:.4f}")
        except Exception as e:
            print(f"Warning: Could not compute ROC AUC for class '{class_name}': {e}")
            continue
    
    if len(roc_auc_scores) == 0:
        raise ValueError("Cannot compute ROC AUC: no catheter line classes have sufficient diversity")
    
    score = np.average(roc_auc_scores)
    print(f"ROC AUC computed for {len(roc_auc_scores)}/{len(CLASSES)} catheter line classes, macro average: {score:.4f}")
    
    return score