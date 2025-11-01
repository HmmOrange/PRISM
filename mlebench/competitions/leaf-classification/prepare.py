import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.competitions.utils import df_to_one_hot
from mlebench.utils import extract, read_csv

from .classes import CLASSES


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    # extract only what we need
    extract(raw / "train.csv.zip", raw)
    extract(raw / "images.zip", raw)

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    new_test_without_labels = new_test.drop(columns=["species"])

    # match the format of the sample submission
    new_test = new_test[["id", "species"]]
    new_test = df_to_one_hot(new_test, "id", "species", classes=CLASSES)

    (public / "train").mkdir(exist_ok=True)
    (public / "test").mkdir(exist_ok=True)

    for file_id in new_train["id"]:
        shutil.copyfile(
            src=raw / "images" / f"{file_id}.jpg",
            dst=public / "train" / f"{file_id}.jpg",
        )

    for file_id in new_test_without_labels["id"]:
        shutil.copyfile(
            src=raw / "images" / f"{file_id}.jpg",
            dst=public / "test" / f"{file_id}.jpg",
        )

    # Check integrity of the files copied
    assert len(new_test_without_labels) == len(
        new_test
    ), "Public and Private tests should have equal length"
    # assert len(list(public.glob("images/*.jpg"))) == len(new_train) + len(
    #     new_test_without_labels
    # ), "Public images should have the same number of images as the sum of train and test"

    # Create a sample submission file
    submission_df = new_test.copy()
    submission_df[CLASSES] = 1 / len(CLASSES)

    # Copy over files
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "answers.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    submission_df.to_csv(public / "sample_submission.csv", index=False)

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of leaf species (99 classes in one-hot format)
    
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
    answers_df = read_csv(private / "answers.csv")  # test with one-hot labels
    
    print(f"Original test samples: {len(answers_df)}")
    
    # Convert one-hot back to single label for stratification
    # Find which column has value 1 for each row (the true species)
    species_columns = [col for col in answers_df.columns if col != 'id']
    true_species = []
    
    for idx, row in answers_df.iterrows():
        # Find the column with value 1
        species_col = None
        for col in species_columns:
            if row[col] == 1:
                species_col = col
                break
        true_species.append(species_col)
    
    answers_df['true_species'] = true_species
    
    # Check distribution of species
    species_counts = pd.Series(true_species).value_counts()
    print(f"Original test species distribution (top 10): {dict(list(species_counts.head(10).items()))}")
    print(f"Total unique species: {len(species_counts)}")
    
    # If test set is already <= max_test_samples, keep original
    if len(answers_df) <= max_test_samples:
        print(f"Test set already has {len(answers_df)} samples (<= {max_test_samples}), keeping original")
        # Copy only private directory
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # Stratified sampling to preserve species distribution
    try:
        _, sampled_test, _, _ = train_test_split(
            answers_df,
            answers_df['true_species'],
            test_size=max_test_samples,
            stratify=answers_df['true_species'],
            random_state=42
        )
    except ValueError as e:
        # If some species have too few samples for stratification, use regular sampling
        print(f"Stratification failed ({e}), using random sampling instead...")
        sampled_test = answers_df.sample(n=max_test_samples, random_state=42)
    
    # Remove the helper column before saving
    sampled_test = sampled_test.drop(columns=['true_species'])
    
    # Log distribution after sampling
    sampled_species = []
    for idx, row in sampled_test.iterrows():
        for col in species_columns:
            if row[col] == 1:
                sampled_species.append(col)
                break
    
    new_species_counts = pd.Series(sampled_species).value_counts()
    print(f"Sampled test species distribution (top 10): {dict(list(new_species_counts.head(10).items()))}")
    print(f"Sampled unique species: {len(new_species_counts)}")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_test.to_csv(lite_private / "answers.csv", index=False)
    
    # Validation
    print("Running validation...")
    assert len(sampled_test) <= max_test_samples, f"Test set too large: {len(sampled_test)}"
    assert sampled_test.shape[1] == len(species_columns) + 1, f"Should have {len(species_columns)} species columns + 1 id column"
    
    print(f"Successfully created lite version with {len(sampled_test)} test samples")
    print(f"Species distribution preserved with {len(new_species_counts)} unique species")
    print(f"Private lite saved to: {lite_private}")