from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.competitions.utils import df_to_one_hot
from mlebench.utils import extract, read_csv

from .classes import CLASSES


def prepare(raw: Path, public: Path, private: Path):
    target_col = "author"
    id_col = "id"

    # extract only what we need
    extract(raw / "train.zip", raw)

    # Create train, test from train split
    old_train = read_csv(raw / "train.csv")
    new_train, new_test = train_test_split(old_train, test_size=0.1, random_state=0)
    new_test_without_labels = new_test.drop(columns=[target_col])

    # private test matches the format of sample submission
    one_hot_new_test = df_to_one_hot(
        new_test.drop(columns=["text"]),
        id_column=id_col,
        target_column=target_col,
        classes=CLASSES,
    )
    # fill the sample submission with arbitrary values (matching kaggle.com)
    sample_submission = one_hot_new_test.copy()
    sample_submission["EAP"] = 0.403493538995863
    sample_submission["HPL"] = 0.287808366106543
    sample_submission["MWS"] = 0.308698094897594

    # save files
    new_train.to_csv(public / "train.csv", index=False)
    new_test_without_labels.to_csv(public / "test.csv", index=False)
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    one_hot_new_test.to_csv(private / "answers.csv", index=False)

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of authors (3 classes in one-hot format)
    
    Authors: EAP (Edgar Allan Poe), HPL (H.P. Lovecraft), MWS (Mary Wollstonecraft Shelley)
    Each sample has exactly one author = 1, others = 0
    
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
    # Find which column has value 1 for each row (the true author)
    author_columns = CLASSES  # ["EAP", "HPL", "MWS"]
    true_authors = []
    
    for idx, row in answers_df.iterrows():
        # Find the column with value 1
        true_author = None
        for col in author_columns:
            if row[col] == 1:
                true_author = col
                break
        true_authors.append(true_author)
    
    answers_df['true_author'] = true_authors
    
    # Check distribution of authors
    author_counts = pd.Series(true_authors).value_counts()
    print(f"Original test author distribution: {author_counts.to_dict()}")
    print(f"Authors: EAP (Edgar Allan Poe), HPL (H.P. Lovecraft), MWS (Mary Wollstonecraft Shelley)")
    
    # If test set is already <= max_test_samples, keep original
    if len(answers_df) <= max_test_samples:
        print(f"Test set already has {len(answers_df)} samples (<= {max_test_samples}), keeping original")
        # Copy only private directory
        import shutil
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # Stratified sampling to preserve author distribution
    try:
        _, sampled_test, _, _ = train_test_split(
            answers_df,
            answers_df['true_author'],
            test_size=max_test_samples,
            stratify=answers_df['true_author'],
            random_state=42
        )
    except ValueError as e:
        # If some authors have too few samples for stratification, use regular sampling
        print(f"Stratification failed ({e}), using random sampling instead...")
        sampled_test = answers_df.sample(n=max_test_samples, random_state=42)
    
    # Remove the helper column before saving
    sampled_test = sampled_test.drop(columns=['true_author'])
    
    # Log distribution after sampling
    sampled_authors = []
    for idx, row in sampled_test.iterrows():
        for col in author_columns:
            if row[col] == 1:
                sampled_authors.append(col)
                break
    
    new_author_counts = pd.Series(sampled_authors).value_counts()
    print(f"Sampled test author distribution: {new_author_counts.to_dict()}")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_test.to_csv(lite_private / "answers.csv", index=False)
    
    # Validation
    print("Running validation...")
    assert len(sampled_test) <= max_test_samples, f"Test set too large: {len(sampled_test)}"
    assert sampled_test.shape[1] == len(author_columns) + 1, f"Should have {len(author_columns)} author columns + 1 id column"
    
    # Verify one-hot encoding: each row should have exactly one 1
    for idx, row in sampled_test.iterrows():
        author_sum = sum(row[col] for col in author_columns)
        assert author_sum == 1, f"Row {idx} should have exactly one author = 1, got sum = {author_sum}"
    
    print(f"Successfully created lite version with {len(sampled_test)} test samples")
    print(f"Author distribution preserved with {len(new_author_counts)} unique authors")
    print(f"Authors: {list(new_author_counts.index)}")
    print(f"Private lite saved to: {lite_private}")