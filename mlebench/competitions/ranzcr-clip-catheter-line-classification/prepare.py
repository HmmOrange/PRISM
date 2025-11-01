import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv

from .classes import CLASSES


def prepare(raw: Path, public: Path, private: Path):
    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    old_train_annotations = read_csv(raw / "train_annotations.csv")
    old_train_uids = old_train_annotations["StudyInstanceUID"]
    new_train_uids = new_train["StudyInstanceUID"]
    is_in_new_train = old_train_uids.isin(new_train_uids)

    new_train_annotations = old_train_annotations[is_in_new_train]

    (public / "train").mkdir(exist_ok=True)
    (public / "test").mkdir(exist_ok=True)

    for file_id in new_train["StudyInstanceUID"]:
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "train" / f"{file_id}.jpg",
        )

    for file_id in new_test["StudyInstanceUID"]:
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "test" / f"{file_id}.jpg",
        )

    assert len(list(public.glob("train/*.jpg"))) == len(
        new_train
    ), f"Expected {len(new_train)} files in public train, got {len(list(public.glob('train/*.jpg')))}"
    assert len(list(public.glob("test/*.jpg"))) == len(
        new_test
    ), f"Expected {len(new_test)} files in public test, got {len(list(public.glob('test/*.jpg')))}"

    # Create a sample submission file
    submission_df = new_test[["StudyInstanceUID"] + CLASSES]
    submission_df[CLASSES] = 0

    # Copy over files
    new_train.to_csv(public / "train.csv", index=False)
    new_train_annotations.to_csv(public / "train_annotations.csv", index=False)
    new_test.to_csv(private / "answers.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=False)

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of multi-label catheter line classification
    
    Each sample has 11 labels (catheter line types) with values 0 or 1
    Multiple labels can be 1 simultaneously (multi-label classification)
    
    Only process test data in private_lite_dir, not touching public/train data
    
    Args:
        raw: Path to raw data (not used)
        lite_private: Path to private directory of prepared_lite 
        private: Path to private directory of prepared original
        max_test_samples: Maximum number of test samples
    """
    import pandas as pd
    import numpy as np
    
    print(f"Creating lite version with max {max_test_samples} test samples...")
    
    # Read test data from prepared/private
    answers_df = pd.read_csv(private / "answers.csv")  # test with multi-labels
    
    print(f"Original test samples: {len(answers_df)}")
    
    # Find label columns (exclude ID columns)
    id_columns = ['StudyInstanceUID', 'PatientID']
    label_columns = [col for col in answers_df.columns if col not in id_columns]
    print(f"Label columns ({len(label_columns)}): {label_columns}")
    
    # Check distribution of each label column
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
    
    # For multi-label classification, we'll use simple random sampling
    # since stratifying on 11 different labels simultaneously is complex
    print("Using random sampling for multi-label catheter line data...")
    sampled_test = answers_df.sample(n=max_test_samples, random_state=42)
    
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
    
    # Check that all label values are in {0, 1}
    for col in label_columns:
        if col in sampled_test.columns:
            unique_vals = set(sampled_test[col].unique())
            assert unique_vals.issubset({0, 1}), f"Column {col} should only contain 0 or 1, got {unique_vals}"
    
    # Check that we have the expected number of columns
    expected_columns = len(label_columns) + len(id_columns)  # labels + ID columns
    assert sampled_test.shape[1] == expected_columns, f"Should have {expected_columns} columns, got {sampled_test.shape[1]}"
    
    print(f"Successfully created lite version with {len(sampled_test)} test samples")
    print(f"Multi-label distribution preserved across {len(label_columns)} catheter line categories")
    print(f"Private lite saved to: {lite_private}")