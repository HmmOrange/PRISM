import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from mlebench.utils import compress, extract, read_csv


def prepare(raw: Path, public: Path, private: Path):

    old_train = read_csv(raw / "train.csv")

    # 4000 / (4000 + 17500) -> test_size is ~0.19
    new_train, new_test = train_test_split(old_train, test_size=0.19, random_state=0)

    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "answers.csv", index=False)

    sample_submission = new_test.copy()
    sample_submission["has_cactus"] = 0.5
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # need to split the train.zip into train.zip and test.zip; to do so need to extract first
    extract(raw / "train.zip", raw)

    # copy
    (public / "train").mkdir(parents=True, exist_ok=True)
    for image_id in tqdm(new_train["id"], desc="Copying train images", total=len(new_train)):
        shutil.copy(raw / "train" / image_id, public / "train" / image_id)

    (public / "test").mkdir(parents=True, exist_ok=True)
    for image_id in tqdm(new_test["id"], desc="Copying test images", total=len(new_test)):
        shutil.copy(raw / "train" / image_id, public / "test" / image_id)

    # # and then recompress
    # compress(public / "train", public / "train.zip")
    # compress(public / "test", public / "test.zip")

    # # and cleanup
    # shutil.rmtree(public / "train")
    # shutil.rmtree(public / "test")
    shutil.rmtree(raw / "train")

    # # checks
    # assert (public / "train.zip").exists(), "public/train.zip should exist"
    # assert (public / "test.zip").exists(), "public/test.zip should exist"
    assert not (raw / "train").exists(), "raw/train/ should not exist"

    # assert len(new_train) + len(new_test) == len(
    #     old_train
    # ), "The lengths of the splits should add up to the original"
    # assert len(new_train) > len(new_test), "The train set should be larger than the test set"
    # assert len(new_test) == len(
    #     sample_submission
    # ), "The test set should match the sample submission"

    # assert (
    #     new_train.columns.tolist()
    #     == new_test.columns.tolist()
    #     == old_train.columns.tolist()
    #     == sample_submission.columns.tolist()
    # ), "All dataframes should have the same columns, i.e. ['id', 'has_cactus']"

    # assert set(new_train["id"]).isdisjoint(new_test["id"]), "Train and test ids should not overlap"
    # assert set(new_test["id"]) == set(
    #     sample_submission["id"]
    # ), "Test and sample_submission ids should match"

    # assert new_train["id"].nunique() == len(new_train), "There should be no duplicate ids in train"
    # assert new_test["id"].nunique() == len(new_test), "There should be no duplicate ids in test"

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of has_cactus (0 and 1)
    
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
    
    # Check distribution of has_cactus in answers_df
    target_counts = answers_df['has_cactus'].value_counts().sort_index()
    print(f"Original test has_cactus distribution: {target_counts.to_dict()}")
    
    # If test set is already <= max_test_samples, keep original
    if len(answers_df) <= max_test_samples:
        print(f"Test set already has {len(answers_df)} samples (<= {max_test_samples}), keeping original")
        # Copy only private directory
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # Stratified sampling to keep distribution (0 and 1)
    _, sampled_test, _, _ = train_test_split(
        answers_df,
        answers_df['has_cactus'],
        test_size=max_test_samples,
        stratify=answers_df['has_cactus'],
        random_state=42
    )
    
    # Log distribution after sampling
    new_target_counts = sampled_test['has_cactus'].value_counts().sort_index()
    print(f"Sampled test has_cactus distribution: {new_target_counts.to_dict()}")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_test.to_csv(lite_private / "answers.csv", index=False)
    
    # Validation
    print("Running validation...")
    assert len(sampled_test) <= max_test_samples, f"Test set too large: {len(sampled_test)}"
    
    print(f"Successfully created lite version with {len(sampled_test)} test samples")
    print(f"Has_cactus distribution preserved: {new_target_counts.to_dict()}")
    print(f"Private lite saved to: {lite_private}")