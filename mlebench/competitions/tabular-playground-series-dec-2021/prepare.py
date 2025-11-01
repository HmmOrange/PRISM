from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    new_test_without_labels = new_test.drop(columns=["Cover_Type"])

    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "answers.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)

    # Create a sample submission file
    submission_df = pd.DataFrame(
        {
            "Id": new_test["Id"].values,
            "Cover_Type": 2,
        }
    )
    submission_df.to_csv(public / "sample_submission.csv", index=False)

    assert len(new_train) + len(new_test) == len(
        old_train
    ), "train and test splits lengths do not add up to original data."
    assert set(new_train["Id"]).isdisjoint(
        set(new_test["Id"])
    ), f"there are Ids in both train and test sets: {set(new_train['Id']).intersection(set(new_test['Id']))}"
    assert len(new_test_without_labels) == len(
        new_test
    ), "public and Private tests should have equal length"

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of Cover_Type (7 classes: 1, 2, 3, 4, 5, 6, 7)
    
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
    
    # Check distribution of Cover_Type
    cover_type_counts = answers_df['Cover_Type'].value_counts().sort_index()
    print(f"Original test Cover_Type distribution: {cover_type_counts.to_dict()}")
    print(f"Cover_Type classes: {sorted(answers_df['Cover_Type'].unique())}")
    
    # If test set is already <= max_test_samples, keep original
    if len(answers_df) <= max_test_samples:
        print(f"Test set already has {len(answers_df)} samples (<= {max_test_samples}), keeping original")
        # Copy only private directory
        import shutil
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # Stratified sampling to preserve Cover_Type distribution
    try:
        _, sampled_test, _, _ = train_test_split(
            answers_df,
            answers_df['Cover_Type'],
            test_size=max_test_samples,
            stratify=answers_df['Cover_Type'],
            random_state=42
        )
    except ValueError as e:
        # If some Cover_Type classes have too few samples for stratification, use regular sampling
        print(f"Stratification failed ({e}), using random sampling instead...")
        sampled_test = answers_df.sample(n=max_test_samples, random_state=42)
    
    # Log distribution after sampling
    new_cover_type_counts = sampled_test['Cover_Type'].value_counts().sort_index()
    print(f"Sampled test Cover_Type distribution: {new_cover_type_counts.to_dict()}")
    print(f"Sampled Cover_Type classes: {sorted(sampled_test['Cover_Type'].unique())}")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_test.to_csv(lite_private / "answers.csv", index=False)
    
    # Validation
    print("Running validation...")
    assert len(sampled_test) <= max_test_samples, f"Test set too large: {len(sampled_test)}"
    assert set(sampled_test['Cover_Type'].unique()).issubset({1, 2, 3, 4, 5, 6, 7}), "Cover_Type should only contain values 1-7"
    assert 'Cover_Type' in sampled_test.columns, "Cover_Type column should exist"
    assert 'Id' in sampled_test.columns, "Id column should exist"
    
    print(f"Successfully created lite version with {len(sampled_test)} test samples")
    print(f"Cover_Type distribution preserved across {len(new_cover_type_counts)} classes")
    print(f"Private lite saved to: {lite_private}")