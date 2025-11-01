import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """

    # Load data
    with open(raw / "train.json") as f:
        old_train = json.load(f)
    with open(raw / "test.json") as f:
        old_test = json.load(f)

    test_ratio = len(old_test) / (len(old_train) + len(old_test))

    all_fields = list([key for key in old_train[0].keys()])
    assert all(set(all_fields) == set([key for key in sample.keys()]) for sample in old_train)
    test_fields = list([key for key in old_test[0].keys()])
    assert all(set(test_fields) == set([key for key in sample.keys()]) for sample in old_test)

    # Create train, test from train split
    new_train, new_test = train_test_split(old_train, test_size=test_ratio, random_state=0)

    new_test_without_labels = []
    for sample in new_test:
        # Keep only the fields in the test set
        sample = {key: sample[key] for key in test_fields}
        new_test_without_labels.append(sample)

    # Save the new train and test splits
    with open(public / "train.json", "w") as f:
        json.dump(new_train, f, indent=4)
    with open(public / "test.json", "w") as f:
        json.dump(new_test_without_labels, f, indent=4)
    # Also save zipped versions "train.json.zip" and "test.json.zip"
    shutil.make_archive(public / "train", "zip", public, "train.json")
    shutil.make_archive(public / "test", "zip", public, "test.json")

    # Create answers
    answers_rows = []
    for sample in new_test:
        answers_rows.append(
            {
                "request_id": sample["request_id"],
                "requester_received_pizza": int(sample["requester_received_pizza"]),
            }
        )
    answers = pd.DataFrame(answers_rows)
    answers.to_csv(private / "answers.csv", index=False)

    # Create sample submission
    sample_submission = answers.copy()
    sample_submission["requester_received_pizza"] = 0
    sample_submission.to_csv(public / "sampleSubmission.csv", index=False)

    # Checks
    assert len(new_train) + len(new_test) == len(
        old_train
    ), f"Expected {len(old_train)} total samples in new_train ({len(new_train)}) and new_test ({len(new_test)})"
    assert len(new_test) == len(
        new_test_without_labels
    ), f"Expected new_test ({len(new_test)}) to have the same length as new_test_without_labels ({len(new_test_without_labels)})"
    assert len(answers) == len(
        new_test
    ), f"Expected answers ({len(answers)}) to have the same length as new_test ({len(new_test)})"
    assert len(sample_submission) == len(
        new_test
    ), f"Expected sample_submission ({len(sample_submission)}) to have the same length as new_test ({len(new_test)})"
    assert set(answers.columns) == set(
        ["request_id", "requester_received_pizza"]
    ), "Answers must have 'request_id' and 'requester_received_pizza' columns"
    assert set(sample_submission.columns) == set(
        ["request_id", "requester_received_pizza"]
    ), "Sample submission must have 'request_id' and 'requester_received_pizza' columns"

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of requester_received_pizza (0: no pizza, 1: received pizza)
    
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
    
    # Check distribution of requester_received_pizza
    target_counts = answers_df['requester_received_pizza'].value_counts().sort_index()
    print(f"Original test requester_received_pizza distribution: {target_counts.to_dict()}")
    print(f"(0: no pizza, 1: received pizza)")
    
    # If test set is already <= max_test_samples, keep original
    if len(answers_df) <= max_test_samples:
        print(f"Test set already has {len(answers_df)} samples (<= {max_test_samples}), keeping original")
        # Copy only private directory
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # Stratified sampling to preserve distribution (0: no pizza, 1: received pizza)
    _, sampled_test, _, _ = train_test_split(
        answers_df,
        answers_df['requester_received_pizza'],
        test_size=max_test_samples,
        stratify=answers_df['requester_received_pizza'],
        random_state=42
    )
    
    # Log distribution after sampling
    new_target_counts = sampled_test['requester_received_pizza'].value_counts().sort_index()
    print(f"Sampled test requester_received_pizza distribution: {new_target_counts.to_dict()}")
    print(f"(0: no pizza, 1: received pizza)")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_test.to_csv(lite_private / "answers.csv", index=False)
    
    # Validation
    print("Running validation...")
    assert len(sampled_test) <= max_test_samples, f"Test set too large: {len(sampled_test)}"
    assert set(sampled_test['requester_received_pizza'].unique()).issubset({0, 1}), "Labels should only be 0 or 1"
    assert set(sampled_test.columns) == {"request_id", "requester_received_pizza"}, "Should have request_id and requester_received_pizza columns"
    
    print(f"Successfully created lite version with {len(sampled_test)} test samples")
    print(f"Pizza distribution preserved: {new_target_counts.to_dict()}")
    print(f"Private lite saved to: {lite_private}")