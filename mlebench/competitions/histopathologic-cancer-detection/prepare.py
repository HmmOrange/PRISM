import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def prepare(raw: Path, public: Path, private: Path):
    old_train = pd.read_csv(raw / "train_labels.csv")

    num_test = len(list((raw / "test").glob("*.tif")))
    test_ratio = num_test / (len(old_train) + num_test)

    new_train_ids, new_test_ids = train_test_split(
        old_train["id"], test_size=test_ratio, random_state=0
    )
    new_train = old_train[old_train["id"].isin(new_train_ids)]
    new_test = old_train[old_train["id"].isin(new_test_ids)]

    assert set(new_train["id"]).isdisjoint(
        set(new_test["id"])
    ), "Train should not contain id's of test images"
    assert len(new_train_ids) + len(new_test_ids) == len(
        old_train
    ), "The combined length of new_train_ids and new_test_ids should equal the length of old_train"

    # Copy over files
    (public / "train").mkdir(exist_ok=True)
    (public / "test").mkdir(exist_ok=True)

    for file_id in tqdm(new_train_ids, desc="Copying train images"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.tif",
            dst=public / "train" / f"{file_id}.tif",
        )
    for file_id in tqdm(new_test_ids, desc="Copying test images"):
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.tif",
            dst=public / "test" / f"{file_id}.tif",
        )

    # Create sample submission
    sample_submission = new_test.copy()
    sample_submission["label"] = 0

    # Copy over files
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "answers.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Check integrity of files copied
    assert len(list(public.glob("train/*.tif"))) == len(
        new_train_ids
    ), "Number of train images should be equal to the number of unique id's in the train set"
    assert len(list(public.glob("test/*.tif"))) == len(
        new_test_ids
    ), "Number of test images should be equal to the number of unique id's in the test set"

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of label (0: no cancer, 1: cancer)
    
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
    answers_df = pd.read_csv(private / "answers.csv")  # test with labels
    
    print(f"Original test samples: {len(answers_df)}")
    
    # Check distribution of label in answers_df
    target_counts = answers_df['label'].value_counts().sort_index()
    print(f"Original test label distribution: {target_counts.to_dict()}")
    print(f"(0: no cancer, 1: cancer)")
    
    # If test set is already <= max_test_samples, keep original
    if len(answers_df) <= max_test_samples:
        print(f"Test set already has {len(answers_df)} samples (<= {max_test_samples}), keeping original")
        # Copy only private directory
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # Stratified sampling to preserve distribution (0: no cancer, 1: cancer)
    _, sampled_test, _, _ = train_test_split(
        answers_df,
        answers_df['label'],
        test_size=max_test_samples,
        stratify=answers_df['label'],
        random_state=42
    )
    
    # Log distribution after sampling
    new_target_counts = sampled_test['label'].value_counts().sort_index()
    print(f"Sampled test label distribution: {new_target_counts.to_dict()}")
    print(f"(0: no cancer, 1: cancer)")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_test.to_csv(lite_private / "answers.csv", index=False)
    
    # Validation
    print("Running validation...")
    assert len(sampled_test) <= max_test_samples, f"Test set too large: {len(sampled_test)}"
    assert set(sampled_test['label'].unique()).issubset({0, 1}), "Labels should only be 0 or 1"
    
    print(f"Successfully created lite version with {len(sampled_test)} test samples")
    print(f"Label distribution preserved: {new_target_counts.to_dict()}")
    print(f"Private lite saved to: {lite_private}")