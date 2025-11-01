from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):

    old_train = read_csv(raw / "train.csv")

    # 900k train, 1.6m - 900k = 700k test; so 700k/1.6m = 0.4375
    # We create our split at 100,000 test samples to get same OOM while keeping as many samples as possible in train
    new_train, new_test = train_test_split(old_train, test_size=100_000, random_state=0)

    # make ids go from 0 to len(new_train) - 1
    new_train.id = np.arange(len(new_train))
    # and from len(new_train) to len(new_train) + len(new_test) - 1
    new_test.id = np.arange(len(new_train), len(new_train) + len(new_test))

    # make downstream files
    new_test_without_labels = new_test.drop(columns=["target"]).copy()
    gold_submission = new_test[["id", "target"]].copy()
    sample_submission = gold_submission.copy()
    sample_submission.target = 0.5

    # save
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "answers.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    gold_submission.to_csv(private / "gold_submission.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # checks
    assert len(new_train) + len(new_test) == len(
        old_train
    ), "Expected the sum of the lengths of the new train and test to be equal to the length of the original train."
    assert len(new_test) == len(
        sample_submission
    ), "Expected the length of the private test to be equal to the length of the sample submission."
    assert len(new_test) == len(
        gold_submission
    ), "Expected the length of the private test to be equal to the length of the gold submission."

    assert (
        new_train.columns.to_list() == old_train.columns.to_list()
    ), "Expected the columns of the new train to be the same as the columns of the original train."
    assert (
        new_test.columns.to_list() == old_train.columns.to_list()
    ), "Expected the columns of the new test to be the same as the columns of the original train"

    # check that ids dont overlap between train and test
    assert set(new_train.id).isdisjoint(
        set(new_test.id)
    ), "Expected the ids of the new train and test to be disjoint."

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of target (0 and 1)
    
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
    answers_df = read_csv(private / "answers.csv")  # test with labels
    
    print(f"Original test samples: {len(answers_df)}")
    
    # Check distribution of target
    target_counts = answers_df['target'].value_counts().sort_index()
    print(f"Original test target distribution: {target_counts.to_dict()}")
    print(f"(0: negative class, 1: positive class)")
    
    # If test set is already <= max_test_samples, keep original
    if len(answers_df) <= max_test_samples:
        print(f"Test set already has {len(answers_df)} samples (<= {max_test_samples}), keeping original")
        # Copy only private directory
        import shutil
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # Stratified sampling to preserve distribution (0 and 1)
    _, sampled_test, _, _ = train_test_split(
        answers_df,
        answers_df['target'],
        test_size=max_test_samples,
        stratify=answers_df['target'],
        random_state=42
    )
    
    # Log distribution after sampling
    new_target_counts = sampled_test['target'].value_counts().sort_index()
    print(f"Sampled test target distribution: {new_target_counts.to_dict()}")
    print(f"(0: negative class, 1: positive class)")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_test.to_csv(lite_private / "answers.csv", index=False)
    
    # Validation
    print("Running validation...")
    assert len(sampled_test) <= max_test_samples, f"Test set too large: {len(sampled_test)}"
    assert set(sampled_test['target'].unique()).issubset({0, 1}), "Target should only be 0 or 1"
    assert 'target' in sampled_test.columns, "target column should exist"
    assert 'id' in sampled_test.columns, "id column should exist"
    
    print(f"Successfully created lite version with {len(sampled_test)} test samples")
    print(f"Target distribution preserved: {new_target_counts.to_dict()}")
    print(f"Private lite saved to: {lite_private}")