import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    new_test_without_labels = new_test.drop(columns=["diagnosis"])

    (public / "test").mkdir(exist_ok=True)
    (public / "train").mkdir(exist_ok=True)

    # Copy data
    for file_id in new_train["id_code"]:
        shutil.copyfile(
            src=raw / "train_images" / f"{file_id}.png",
            dst=public / "train" / f"{file_id}.png",
        )

    for file_id in new_test_without_labels["id_code"]:
        shutil.copyfile(
            src=raw / "train_images" / f"{file_id}.png",
            dst=public / "test" / f"{file_id}.png",
        )

    # Check integrity of the files copied
    assert set(new_train["id_code"]).isdisjoint(
        set(new_test["id_code"])
    ), "Train and test sets should have no shared ids"

    assert len(new_test_without_labels) == len(
        new_test
    ), "Public and Private tests should have equal length"

    assert len(list(public.glob("train/*.png"))) == len(
        new_train
    ), "Public train images should have the same number of images as the length of train set"

    assert len(list(public.glob("test/*.png"))) == len(
        new_test_without_labels
    ), "Public test images should have the same number of images as the length of test set"

    train_image_files = set(public.glob("train/*.png"))
    test_image_files = set(public.glob("test/*.png"))
    common_files = train_image_files.intersection(test_image_files)
    assert not common_files, f"Images found in both train and test: {common_files}"

    for file_id in new_test["id_code"]:
        assert (
            public / "test" / f"{file_id}.png"
        ).exists(), f"Image file for {file_id} not found in test"

    for file_id in new_train["id_code"]:
        assert (
            public / "train" / f"{file_id}.png"
        ).exists(), f"Image file for {file_id} not found in train"

    # Create a sample submission file
    submission_df = new_test.copy()
    submission_df["diagnosis"] = 0
    # Write CSVs
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "answers.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=False)

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of diagnosis (0, 1, 2, 3, 4)
    
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
    
    # Check distribution of diagnosis in answers_df
    target_counts = answers_df['diagnosis'].value_counts().sort_index()
    print(f"Original test diagnosis distribution: {target_counts.to_dict()}")
    
    # If test set is already <= max_test_samples, keep original
    if len(answers_df) <= max_test_samples:
        print(f"Test set already has {len(answers_df)} samples (<= {max_test_samples}), keeping original")
        # Copy only private directory
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # Stratified sampling to preserve distribution (0, 1, 2, 3, 4)
    _, sampled_test, _, _ = train_test_split(
        answers_df,
        answers_df['diagnosis'],
        test_size=max_test_samples,
        stratify=answers_df['diagnosis'],
        random_state=42
    )
    
    # Log distribution after sampling
    new_target_counts = sampled_test['diagnosis'].value_counts().sort_index()
    print(f"Sampled test diagnosis distribution: {new_target_counts.to_dict()}")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_test.to_csv(lite_private / "answers.csv", index=False)
    
    # Validation
    print("Running validation...")
    assert len(sampled_test) <= max_test_samples, f"Test set too large: {len(sampled_test)}"
    
    print(f"Successfully created lite version with {len(sampled_test)} test samples")
    print(f"Diagnosis distribution preserved: {new_target_counts.to_dict()}")
    print(f"Private lite saved to: {lite_private}")