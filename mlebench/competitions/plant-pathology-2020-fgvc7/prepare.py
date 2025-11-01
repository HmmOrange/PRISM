import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from mlebench.utils import read_csv


def prepare(raw: Path, public: Path, private: Path) -> None:
    expected_train_columns = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    expected_test_columns = ["image_id"]
    expected_answers_columns = expected_train_columns
    expected_sample_submission_columns = expected_train_columns

    old_train = read_csv(raw / "train.csv")
    new_train, answers = train_test_split(old_train, test_size=0.1, random_state=0)

    assert set(new_train.columns) == set(
        expected_train_columns
    ), f"Expected `new_train` to have columns {expected_train_columns} but got {new_train.columns}"

    assert set(answers.columns) == set(
        expected_answers_columns
    ), f"Expected `answers` to have columns {expected_answers_columns} but got {answers.columns}"

    new_train_image_ids = new_train["image_id"].unique()
    new_test_image_ids = answers["image_id"].unique()
    to_new_image_id = {
        **{old_id: f"Train_{i}" for i, old_id in enumerate(new_train_image_ids)},
        **{old_id: f"Test_{i}" for i, old_id in enumerate(new_test_image_ids)},
    }

    new_train["image_id"] = new_train["image_id"].replace(to_new_image_id)
    answers["image_id"] = answers["image_id"].replace(to_new_image_id)

    new_test = answers[["image_id"]].copy()

    assert set(new_test.columns) == set(
        expected_test_columns
    ), f"Expected `new_test` to have columns {expected_test_columns} but got {new_test.columns}"

    sample_submission = answers[["image_id"]].copy()
    sample_submission[["healthy", "multiple_diseases", "rust", "scab"]] = 0.25

    assert set(sample_submission.columns) == set(
        expected_sample_submission_columns
    ), f"Expected `sample_submission` to have columns {expected_sample_submission_columns} but got {sample_submission.columns}"

    private.mkdir(exist_ok=True, parents=True)
    public.mkdir(exist_ok=True, parents=True)
    (public / "images").mkdir(exist_ok=True)

    for old_image_id in tqdm(old_train["image_id"], desc="Copying over train & test images"):
        assert old_image_id.startswith(
            "Train_"
        ), f"Expected train image id `{old_image_id}` to start with `Train_`."

        new_image_id = to_new_image_id.get(old_image_id, old_image_id)

        assert (
            raw / "images" / f"{old_image_id}.jpg"
        ).exists(), f"Image `{old_image_id}.jpg` does not exist in `{raw / 'images'}`."

        shutil.copyfile(
            src=raw / "images" / f"{old_image_id}.jpg",
            dst=public / "images" / f"{new_image_id}.jpg",
        )

    answers.to_csv(private / "answers.csv", index=False)

    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    new_test.to_csv(public / "test.csv", index=False)
    new_train.to_csv(public / "train.csv", index=False)

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of plant pathology classes (4 classes in one-hot format)
    
    Classes: healthy, multiple_diseases, rust, scab
    Each sample has exactly one class = 1, others = 0
    
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
    # Find which column has value 1 for each row (the true class)
    class_columns = ["healthy", "multiple_diseases", "rust", "scab"]
    true_classes = []
    
    for idx, row in answers_df.iterrows():
        # Find the column with value 1
        true_class = None
        for col in class_columns:
            if row[col] == 1:
                true_class = col
                break
        true_classes.append(true_class)
    
    answers_df['true_class'] = true_classes
    
    # Check distribution of classes
    class_counts = pd.Series(true_classes).value_counts()
    print(f"Original test class distribution: {class_counts.to_dict()}")
    
    # If test set is already <= max_test_samples, keep original
    if len(answers_df) <= max_test_samples:
        print(f"Test set already has {len(answers_df)} samples (<= {max_test_samples}), keeping original")
        # Copy only private directory
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # Stratified sampling to preserve class distribution
    try:
        _, sampled_test, _, _ = train_test_split(
            answers_df,
            answers_df['true_class'],
            test_size=max_test_samples,
            stratify=answers_df['true_class'],
            random_state=42
        )
    except ValueError as e:
        # If some classes have too few samples for stratification, use regular sampling
        print(f"Stratification failed ({e}), using random sampling instead...")
        sampled_test = answers_df.sample(n=max_test_samples, random_state=42)
    
    # Remove the helper column before saving
    sampled_test = sampled_test.drop(columns=['true_class'])
    
    # Log distribution after sampling
    sampled_classes = []
    for idx, row in sampled_test.iterrows():
        for col in class_columns:
            if row[col] == 1:
                sampled_classes.append(col)
                break
    
    new_class_counts = pd.Series(sampled_classes).value_counts()
    print(f"Sampled test class distribution: {new_class_counts.to_dict()}")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_test.to_csv(lite_private / "answers.csv", index=False)
    
    # Validation
    print("Running validation...")
    assert len(sampled_test) <= max_test_samples, f"Test set too large: {len(sampled_test)}"
    assert sampled_test.shape[1] == len(class_columns) + 1, f"Should have {len(class_columns)} class columns + 1 image_id column"
    
    # Verify one-hot encoding: each row should have exactly one 1
    for idx, row in sampled_test.iterrows():
        class_sum = sum(row[col] for col in class_columns)
        assert class_sum == 1, f"Row {idx} should have exactly one class = 1, got sum = {class_sum}"
    
    print(f"Successfully created lite version with {len(sampled_test)} test samples")
    print(f"Plant pathology class distribution preserved with {len(new_class_counts)} unique classes")
    print(f"Classes: {list(new_class_counts.index)}")
    print(f"Private lite saved to: {lite_private}")