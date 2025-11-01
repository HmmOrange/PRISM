from pathlib import Path

from mlebench.utils import extract, read_csv

from .classes import CLASSES


def prepare(raw: Path, public: Path, private: Path):
    # extract only what is needed
    extract(raw / "train.csv.zip", raw)
    extract(raw / "test.csv.zip", raw)
    extract(raw / "test_labels.csv.zip", raw)
    extract(raw / "sample_submission.csv.zip", raw)

    # the test set is provided, so we dont have to split the train set nor form the sample submission
    train_with_labels = read_csv(raw / "train.csv")
    test_without_labels = read_csv(raw / "test.csv")
    answers = read_csv(raw / "test_labels.csv")
    sample_submission = read_csv(raw / "sample_submission.csv")
    sample_submission[CLASSES] = 0.5

    # save to public
    train_with_labels.to_csv(public / "train.csv", index=False)
    test_without_labels.to_csv(public / "test.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # save to private
    answers.to_csv(private / "answers.csv", index=False)

    assert len(answers) == len(
        sample_submission
    ), "Private test set and sample submission should be of the same length"

    assert sorted(answers["id"]) == sorted(
        test_without_labels["id"]
    ), "Private and Public test IDs should match"
    assert sorted(sample_submission["id"]) == sorted(
        test_without_labels["id"]
    ), "Public test and sample submission IDs should match"
    assert (
        len(set(train_with_labels["id"]) & set(test_without_labels["id"])) == 0
    ), "Train and test IDs should not overlap"

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of multi-label toxic comments
    
    Each sample has 6 labels (toxic, severe_toxic, obscene, threat, insult, identity_hate)
    with values: 0 (not toxic), 1 (toxic), or -1 (unknown/not evaluated)
    
    Only process test data in private_lite_dir, not touching public/train data
    
    Args:
        raw: Path to raw data (not used)
        lite_private: Path to private directory of prepared_lite 
        private: Path to private directory of prepared original
        max_test_samples: Maximum number of test samples
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    print(f"Creating lite version with max {max_test_samples} test samples...")
    
    # Read test data from prepared/private
    answers_df = read_csv(private / "answers.csv")  # test with multi-labels
    
    print(f"Original test samples: {len(answers_df)}")
    
    # Check distribution of each label column
    label_columns = CLASSES  # toxic, severe_toxic, obscene, threat, insult, identity_hate
    print(f"Label columns: {label_columns}")
    
    for col in label_columns:
        if col in answers_df.columns:
            col_counts = answers_df[col].value_counts().sort_index()
            print(f"  {col}: {col_counts.to_dict()}")
    
    # If test set is already <= max_test_samples, keep original
    if len(answers_df) <= max_test_samples:
        print(f"Test set already has {len(answers_df)} samples (<= {max_test_samples}), keeping original")
        # Copy only private directory
        import shutil
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # For multi-label classification with -1 values, we need to be careful
    # First, identify rows that will be kept after filtering (sum >= 0)
    keep_mask = answers_df[label_columns].sum(axis=1) >= 0
    valid_rows = answers_df[keep_mask]
    
    print(f"Valid rows after -1 filtering: {len(valid_rows)} out of {len(answers_df)}")
    
    # Check if we have enough valid rows
    if len(valid_rows) <= max_test_samples:
        print(f"Using all {len(valid_rows)} valid rows (after -1 filtering)")
        sampled_test = valid_rows
    else:
        print("Using random sampling from valid rows (no -1 values)...")
        sampled_test = valid_rows.sample(n=max_test_samples, random_state=42)
    
    # Ensure we have diversity in each label column for ROC AUC computation
    print("Checking label diversity for ROC AUC computation...")
    for col in label_columns:
        if col in sampled_test.columns:
            unique_vals = sampled_test[col].unique()
            if len(unique_vals) <= 1:
                print(f"⚠️  WARNING: Column '{col}' has only {len(unique_vals)} unique value(s)")
                print(f"   This may cause ROC AUC computation to return NaN")
    
    # Log distribution after sampling
    print(f"Sampled test samples: {len(sampled_test)}")
    for col in label_columns:
        if col in sampled_test.columns:
            col_counts = sampled_test[col].value_counts().sort_index()
            print(f"  {col}: {col_counts.to_dict()}")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_test.to_csv(lite_private / "answers.csv", index=False)
    
    # Validation
    print("Running validation...")
    assert len(sampled_test) <= max_test_samples, f"Test set too large: {len(sampled_test)}"
    
    # Check that all label values are in {-1, 0, 1}
    for col in label_columns:
        if col in sampled_test.columns:
            unique_vals = set(sampled_test[col].unique())
            assert unique_vals.issubset({-1, 0, 1}), f"Column {col} should only contain -1, 0, or 1, got {unique_vals}"
    
    print(f"Successfully created lite version with {len(sampled_test)} test samples")
    print(f"Multi-label distribution preserved across {len(label_columns)} toxic comment categories")
    print(f"Private lite saved to: {lite_private}")