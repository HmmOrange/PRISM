import glob
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlebench.utils import extract, read_csv


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    # Extract only what we need
    extract(raw / "train.zip", raw / "train")
    extract(raw / "train.csv.zip", raw / "train.csv")
    extract(raw / "test.zip", raw / "test")
    extract(raw / "test.csv.zip", raw / "test.csv")

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv/train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)

    # Make ids go 1, 2, ... for both train and test. Keep old ids so we can map ids of other files
    old_train_id_to_new = {
        old_id: new_id for new_id, old_id in enumerate(new_train["id"], start=1)
    }  # id starts from 1
    new_train["id"] = new_train["id"].map(old_train_id_to_new)

    old_test_id_to_new = {
        old_id: new_id for new_id, old_id in enumerate(new_test["id"], start=1)
    }  # id starts from 1
    new_test["id"] = new_test["id"].map(old_test_id_to_new)

    new_test_without_labels = new_test.drop(
        columns=["formation_energy_ev_natom", "bandgap_energy_ev"]
    )

    # Copy over files
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "answers.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)

    train_paths = sorted(glob.glob(str(raw / "train/train/**/*.xyz")))
    for src in train_paths:
        id = int(Path(src).parts[-2])
        if id not in old_train_id_to_new.keys():  # Filter for train ids
            continue

        new_id = old_train_id_to_new[id]
        (public / "train" / str(new_id)).mkdir(parents=True, exist_ok=True)
        shutil.copy(src=src, dst=public / "train" / str(new_id) / "geometry.xyz")
    assert len(list(public.glob("train/**/*.xyz"))) == len(
        new_train
    ), f"Expected {len(new_train)} train geometry files, found {len(list(public.glob('train/**/*.xyz')))}"

    for src in train_paths:
        id = int(Path(src).parts[-2])
        if id not in old_test_id_to_new.keys():  # Filter for test ids
            continue

        new_id = old_test_id_to_new[id]
        (public / "test" / str(new_id)).mkdir(parents=True, exist_ok=True)
        shutil.copy(src=src, dst=public / "test" / str(new_id) / "geometry.xyz")
    assert len(list(public.glob("test/**/*.xyz"))) == len(
        new_test
    ), f"Expected {len(new_test)} test geometry files, found {len(list(public.glob('test/**/*.xyz')))}"

    # Create mock submission
    sample_submission = pd.DataFrame(
        {"id": new_test["id"], "formation_energy_ev_natom": 0.1779, "bandgap_energy_ev": 1.8892}
    )
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    assert len(sample_submission) == len(
        new_test
    ), "Sample submission should have the same number of rows as the test set"

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of both regression targets:
    - formation_energy_ev_natom
    - bandgap_energy_ev
    
    For multi-target regression, we use quantile-based sampling on the first target
    
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
    
    # Check distribution of both targets
    target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    
    for target_col in target_cols:
        if target_col in answers_df.columns:
            target_stats = answers_df[target_col].describe()
            print(f"Original {target_col} distribution:")
            print(f"  Mean: {target_stats['mean']:.4f}")
            print(f"  Std: {target_stats['std']:.4f}")
            print(f"  Min: {target_stats['min']:.4f}")
            print(f"  25%: {target_stats['25%']:.4f}")
            print(f"  50%: {target_stats['50%']:.4f}")
            print(f"  75%: {target_stats['75%']:.4f}")
            print(f"  Max: {target_stats['max']:.4f}")
    
    # If test set is already <= max_test_samples, keep original
    if len(answers_df) <= max_test_samples:
        print(f"Test set already has {len(answers_df)} samples (<= {max_test_samples}), keeping original")
        # Copy only private directory
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # Quantile-based sampling on the first target (formation_energy_ev_natom)
    print("Using quantile-based sampling on formation_energy_ev_natom for multi-target regression...")
    primary_target = "formation_energy_ev_natom"
    
    # Create quantile bins
    n_bins = min(10, max_test_samples // 5)  # At least 5 samples per bin
    answers_df['quantile_bin'] = pd.qcut(answers_df[primary_target], q=n_bins, labels=False, duplicates='drop')
    
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
    
    # Log distribution after sampling for both targets
    for target_col in target_cols:
        if target_col in sampled_test.columns:
            target_stats = sampled_test[target_col].describe()
            print(f"Sampled {target_col} distribution:")
            print(f"  Mean: {target_stats['mean']:.4f}")
            print(f"  Std: {target_stats['std']:.4f}")
            print(f"  Min: {target_stats['min']:.4f}")
            print(f"  25%: {target_stats['25%']:.4f}")
            print(f"  50%: {target_stats['50%']:.4f}")
            print(f"  75%: {target_stats['75%']:.4f}")
            print(f"  Max: {target_stats['max']:.4f}")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_test.to_csv(lite_private / "answers.csv", index=False)
    
    # Validation
    print("Running validation...")
    assert len(sampled_test) <= max_test_samples, f"Test set too large: {len(sampled_test)}"
    
    for target_col in target_cols:
        assert target_col in sampled_test.columns, f"{target_col} column should exist"
        assert sampled_test[target_col].notna().all(), f"All {target_col} values should be non-null"
    
    print(f"Successfully created lite version with {len(sampled_test)} test samples")
    print(f"Multi-target regression distribution preserved using quantile-based sampling")
    print(f"Primary target for sampling: {primary_target}")
    print(f"Private lite saved to: {lite_private}")