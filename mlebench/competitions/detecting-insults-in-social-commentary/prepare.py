import shutil
from pathlib import Path

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    # the test set labels are provided so we can just copy things without doing the split ourselves
    shutil.copy(raw / "train.csv", public / "train.csv")
    shutil.copy(raw / "test_with_solutions.csv", private / "test.csv")

    # the public test currently online is for the verification stage, which we are ignoring
    # (we are ignoring because there is some train/test leakage and unclear how this factors in leaderboard)
    # we can recover the original public test set by removing the labels from the private test set
    # can make the gold submission while we're at it
    gold_submission = read_csv(private / "test.csv")
    gold_submission = gold_submission[["Insult", "Date", "Comment"]]
    gold_submission.to_csv(private / "answers.csv", index=False)

    public_test = gold_submission.drop(columns=["Insult"]).copy()
    public_test.to_csv(public / "test.csv", index=False)

    # finally, we also make our own sample_submission, same reasons as public test
    # but match the format of what's online
    sample_submission = gold_submission.copy()
    sample_submission["Insult"] = 0
    sample_submission.to_csv(public / "sample_submission_null.csv", index=False)

    # checks
    public_train = read_csv(public / "train.csv")
    public_test = read_csv(public / "test.csv")
    private_test = read_csv(private / "test.csv")

    # no `Id` column in train, so we check comment content instead
    assert public_train.columns.to_list() == [
        "Insult",
        "Date",
        "Comment",
    ], "Train columns should be Insult, Date, Comment"
    assert public_test.columns.to_list() == [
        "Date",
        "Comment",
    ], "Test columns should be Date, Comment"
    assert sample_submission.columns.to_list() == [
        "Insult",
        "Date",
        "Comment",
    ], "Sample submission columns should be Insult, Date, Comment"
    assert gold_submission.columns.to_list() == [
        "Insult",
        "Date",
        "Comment",
    ], "Gold submission columns should be Insult, Date, Comment"
    assert private_test.columns.to_list() == [
        "Insult",
        "Date",
        "Comment",
        "Usage",
    ], "Private test columns should be Insult, Date, Comment, Usage"

    assert set(public_train["Comment"]).isdisjoint(
        set(public_test["Comment"])
    ), "None of the test comments should be in the train comments"
    assert public_test.equals(
        private_test.drop(columns=["Insult", "Usage"])
    ), "Public test should be identical to private test, modulo the Insult and Usage columns"
    assert set(public_test["Comment"]) == set(
        sample_submission["Comment"]
    ), "Public test and sample submission should have the same Comments"
    assert set(public_test["Comment"]) == set(
        gold_submission["Comment"]
    ), "Public test and gold submission should have the same Comments"

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of Insult (0 and 1)
    
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
    
    # Check distribution of Insult in answers_df
    target_counts = answers_df['Insult'].value_counts().sort_index()
    print(f"Original test Insult distribution: {target_counts.to_dict()}")
    
    # If test set is already <= max_test_samples, keep original
    if len(answers_df) <= max_test_samples:
        print(f"Test set already has {len(answers_df)} samples (<= {max_test_samples}), keeping original")
        # Copy only private directory
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # Stratified sampling to preserve distribution (0 and 1)
    _, sampled_test, _, _ = train_test_split(
        answers_df,
        answers_df['Insult'],
        test_size=max_test_samples,
        stratify=answers_df['Insult'],
        random_state=42
    )
    
    # Log distribution after sampling
    new_target_counts = sampled_test['Insult'].value_counts().sort_index()
    print(f"Sampled test Insult distribution: {new_target_counts.to_dict()}")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_test.to_csv(lite_private / "answers.csv", index=False)
    
    # Validation
    print("Running validation...")
    assert len(sampled_test) <= max_test_samples, f"Test set too large: {len(sampled_test)}"
    
    print(f"Successfully created lite version with {len(sampled_test)} test samples")
    print(f"Insult distribution preserved: {new_target_counts.to_dict()}")
    print(f"Private lite saved to: {lite_private}")