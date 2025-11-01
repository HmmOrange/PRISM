import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.competitions.utils import df_to_one_hot
from mlebench.utils import read_csv

from .dogs import DOGS_LIST


def to_one_hot(df):
    return df_to_one_hot(df, id_column="id", target_column="breed", classes=DOGS_LIST)


def prepare(raw: Path, public: Path, private: Path):

    # Create train, test from train split
    old_train = read_csv(raw / "labels.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    # one hot the private test because we will one-hot the submission, as per kaggle.com
    new_test = to_one_hot(new_test)

    # Copy over files
    new_train.to_csv(public / "train.csv", index=False)
    new_test.to_csv(private / "answers.csv", index=False)

    (public / "train").mkdir(exist_ok=True)

    for file_id in new_train["id"]:
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "train" / f"{file_id}.jpg",
        )

    (public / "test").mkdir(exist_ok=True)

    for file_id in new_test["id"]:
        shutil.copyfile(
            src=raw / "train" / f"{file_id}.jpg",
            dst=public / "test" / f"{file_id}.jpg",
        )

    # Check integrity of the files copied
    assert len(list(public.glob("train/*.jpg"))) == len(new_train)
    assert len(list(public.glob("test/*.jpg"))) == len(new_test)

    # Create a sample submission file
    submission_df = new_test.copy()
    for col in submission_df.columns[1:]:
        submission_df[col] = submission_df[col].astype("float64")
    submission_df.iloc[:, 1:] = 1 / 120
    submission_df.to_csv(public / "sample_submission.csv", index=False)

    assert submission_df.shape == (len(new_test), 121)  # 1 id column + 120 breeds

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of dog breeds (120 classes in one-hot format)
    
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
    # Find which column has value 1 for each row (the true breed)
    breed_columns = [col for col in answers_df.columns if col != 'id']
    true_breeds = []
    
    for idx, row in answers_df.iterrows():
        # Find the column with value 1
        breed_col = None
        for col in breed_columns:
            if row[col] == 1:
                breed_col = col
                break
        true_breeds.append(breed_col)
    
    answers_df['true_breed'] = true_breeds
    
    # Check distribution of breeds
    breed_counts = pd.Series(true_breeds).value_counts()
    print(f"Original test breed distribution (top 10): {dict(list(breed_counts.head(10).items()))}")
    print(f"Total unique breeds: {len(breed_counts)}")
    
    # If test set is already <= max_test_samples, keep original
    if len(answers_df) <= max_test_samples:
        print(f"Test set already has {len(answers_df)} samples (<= {max_test_samples}), keeping original")
        # Copy only private directory
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # Stratified sampling to preserve breed distribution
    try:
        _, sampled_test, _, _ = train_test_split(
            answers_df,
            answers_df['true_breed'],
            test_size=max_test_samples,
            stratify=answers_df['true_breed'],
            random_state=42
        )
    except ValueError as e:
        # If some breeds have too few samples for stratification, use regular sampling
        print(f"Stratification failed ({e}), using random sampling instead...")
        sampled_test = answers_df.sample(n=max_test_samples, random_state=42)
    
    # Remove the helper column before saving
    sampled_test = sampled_test.drop(columns=['true_breed'])
    
    # Log distribution after sampling
    sampled_breeds = []
    for idx, row in sampled_test.iterrows():
        for col in breed_columns:
            if row[col] == 1:
                sampled_breeds.append(col)
                break
    
    new_breed_counts = pd.Series(sampled_breeds).value_counts()
    print(f"Sampled test breed distribution (top 10): {dict(list(new_breed_counts.head(10).items()))}")
    print(f"Sampled unique breeds: {len(new_breed_counts)}")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_test.to_csv(lite_private / "answers.csv", index=False)
    
    # Validation
    print("Running validation...")
    assert len(sampled_test) <= max_test_samples, f"Test set too large: {len(sampled_test)}"
    assert sampled_test.shape[1] == len(breed_columns) + 1, f"Should have {len(breed_columns)} breed columns + 1 id column"
    
    print(f"Successfully created lite version with {len(sampled_test)} test samples")
    print(f"Breed distribution preserved with {len(new_breed_counts)} unique breeds")
    print(f"Private lite saved to: {lite_private}")