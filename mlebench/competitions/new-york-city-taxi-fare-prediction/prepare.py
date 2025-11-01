import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path):
    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")

    # Train is c. 55M rows, original test is 9914 rows
    new_train, new_test = train_test_split(old_train, test_size=9914, random_state=0)
    new_test_without_labels = new_test.drop(columns=["fare_amount"])

    # Create a sample submission file
    submission_df = new_test.copy()[["key"]]
    submission_df["fare_amount"] = 11.35

    # Write CSVs
    new_train.to_csv(public / "train.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=False)
    new_test.to_csv(private / "answers.csv", index=False)

    # Copy over other files
    shutil.copy(raw / "GCP-Coupons-Instructions.rtf", public / "GCP-Coupons-Instructions.rtf")

    # Checks
    assert set(new_train["key"]).isdisjoint(
        set(new_test["key"])
    ), "Train and test sets share samples!"
    assert new_test.shape[1] == 8, f"Test set should have 8 columns, but has {new_test.shape[1]}"
    assert (
        new_test_without_labels.shape[1] == 7
    ), f"Test set without labels should have 7 columns, but has {new_test_without_labels.shape[1]}"
    assert new_train.shape[1] == 8, f"Train set should have 8 columns, but has {new_train.shape[1]}"
    assert (
        submission_df.shape[1] == 2
    ), f"Sample submission should have 2 columns, but has {submission_df.shape[1]}"
    assert (
        submission_df.shape[0] == new_test.shape[0]
    ), f"Sample submission should have {new_test.shape[0]} rows, but has {submission_df.shape[0]}"

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of fare_amount (regression target)
    
    For regression, we use quantile-based sampling to maintain the distribution
    
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
    answers_df = read_csv(private / "answers.csv")  # test with regression targets
    
    print(f"Original test samples: {len(answers_df)}")
    
    # Check distribution of fare_amount
    fare_stats = answers_df['fare_amount'].describe()
    print(f"Original fare_amount distribution:")
    print(f"  Mean: ${fare_stats['mean']:.2f}")
    print(f"  Std: ${fare_stats['std']:.2f}")
    print(f"  Min: ${fare_stats['min']:.2f}")
    print(f"  25%: ${fare_stats['25%']:.2f}")
    print(f"  50%: ${fare_stats['50%']:.2f}")
    print(f"  75%: ${fare_stats['75%']:.2f}")
    print(f"  Max: ${fare_stats['max']:.2f}")
    
    # If test set is already <= max_test_samples, keep original
    if len(answers_df) <= max_test_samples:
        print(f"Test set already has {len(answers_df)} samples (<= {max_test_samples}), keeping original")
        # Copy only private directory
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # Quantile-based sampling to preserve distribution
    print("Using quantile-based sampling for regression target...")
    
    # Create quantile bins (e.g., 10 bins for deciles)
    n_bins = min(10, max_test_samples // 5)  # At least 5 samples per bin
    answers_df['quantile_bin'] = pd.qcut(answers_df['fare_amount'], q=n_bins, labels=False, duplicates='drop')
    
    # Sample proportionally from each bin
    sampled_dfs = []
    for bin_id in sorted(answers_df['quantile_bin'].unique()):
        bin_data = answers_df[answers_df['quantile_bin'] == bin_id]
        bin_size = len(bin_data)
        
        # Calculate samples for this bin (proportional)
        bin_samples = max(1, int(max_test_samples * bin_size / len(answers_df)))
        bin_samples = min(bin_samples, bin_size)  # Don't exceed bin size
        
        # Sample from this bin
        if bin_samples < bin_size:
            sampled_bin = bin_data.sample(n=bin_samples, random_state=42)
        else:
            sampled_bin = bin_data
        
        sampled_dfs.append(sampled_bin)
        print(f"  Bin {bin_id}: {bin_size} -> {len(sampled_bin)} samples")
    
    # Combine all sampled data
    sampled_test = pd.concat(sampled_dfs, ignore_index=True)
    
    # If we have too many samples, randomly remove some
    if len(sampled_test) > max_test_samples:
        sampled_test = sampled_test.sample(n=max_test_samples, random_state=42)
    
    # Remove helper column
    sampled_test = sampled_test.drop(columns=['quantile_bin'])
    
    # Log distribution after sampling
    sampled_fare_stats = sampled_test['fare_amount'].describe()
    print(f"Sampled fare_amount distribution:")
    print(f"  Mean: ${sampled_fare_stats['mean']:.2f}")
    print(f"  Std: ${sampled_fare_stats['std']:.2f}")
    print(f"  Min: ${sampled_fare_stats['min']:.2f}")
    print(f"  25%: ${sampled_fare_stats['25%']:.2f}")
    print(f"  50%: ${sampled_fare_stats['50%']:.2f}")
    print(f"  75%: ${sampled_fare_stats['75%']:.2f}")
    print(f"  Max: ${sampled_fare_stats['max']:.2f}")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_test.to_csv(lite_private / "answers.csv", index=False)
    
    # Validation
    print("Running validation...")
    assert len(sampled_test) <= max_test_samples, f"Test set too large: {len(sampled_test)}"
    assert 'fare_amount' in sampled_test.columns, "fare_amount column should exist"
    assert sampled_test['fare_amount'].notna().all(), "All fare_amount values should be non-null"
    
    print(f"Successfully created lite version with {len(sampled_test)} test samples")
    print(f"Fare distribution preserved using quantile-based sampling")
    print(f"Private lite saved to: {lite_private}")