import csv
import zipfile
from pathlib import Path

from sklearn.model_selection import train_test_split

from mlebench.utils import compress, extract, read_csv


def prepare(raw: Path, public: Path, private: Path):

    # Extract
    extract(raw / "en_test_2.csv.zip", raw)  # We only use the 2nd stage test set
    extract(raw / "en_train.csv.zip", raw)
    extract(raw / "en_sample_submission_2.csv.zip", raw)

    # Create train and test splits from train set
    old_train = read_csv(raw / "en_train.csv")

    # We split so that we don't share any sentence_ids between train and test
    # This gives us len(new_train) = 8924976 and len(answers) = 993465
    # Original was len(old_train) = 9918441 and len(old_test) = 956046
    unique_sentence_ids = old_train["sentence_id"].unique()
    train_sentence_ids, test_sentence_ids = train_test_split(
        unique_sentence_ids, test_size=0.1, random_state=0
    )
    new_train = old_train[old_train["sentence_id"].isin(train_sentence_ids)]
    answers = old_train[old_train["sentence_id"].isin(test_sentence_ids)]
    assert set(new_train["sentence_id"]).isdisjoint(
        set(answers["sentence_id"])
    ), f"sentence_id is not disjoint between train and test sets"

    # "sentence_id" counts need to be reset for new_train and answers
    new_train_id_mapping = {
        old_id: new_id for new_id, old_id in enumerate(new_train["sentence_id"].unique())
    }
    new_train.loc[:, "sentence_id"] = new_train["sentence_id"].map(new_train_id_mapping)
    answers_id_mapping = {
        old_id: new_id for new_id, old_id in enumerate(answers["sentence_id"].unique())
    }
    answers.loc[:, "sentence_id"] = answers["sentence_id"].map(answers_id_mapping)

    # Create new test set
    new_test = answers.drop(["after", "class"], axis=1).copy()

    # Reformat answers to match sample submission format
    answers = answers[["sentence_id", "token_id", "after"]].copy()
    answers["id"] = answers["sentence_id"].astype(str) + "_" + answers["token_id"].astype(str)
    answers = answers[["id", "after"]]

    # Create sample submission
    sample_submission = new_test[["sentence_id", "token_id", "before"]].copy()
    sample_submission["id"] = (
        sample_submission["sentence_id"].astype(str)
        + "_"
        + sample_submission["token_id"].astype(str)
    )
    sample_submission["after"] = sample_submission["before"]
    sample_submission = sample_submission[["id", "after"]]

    # Checks
    assert new_train.columns.tolist() == [
        "sentence_id",
        "token_id",
        "class",
        "before",
        "after",
    ], f"new_train.columns.tolist() == {new_train.columns.tolist()}"
    assert new_test.columns.tolist() == [
        "sentence_id",
        "token_id",
        "before",
    ], f"new_test.columns.tolist() == {new_test.columns.tolist()}"
    assert sample_submission.columns.tolist() == [
        "id",
        "after",
    ], f"sample_submission.columns.tolist() == {sample_submission.columns.tolist()}"
    assert answers.columns.tolist() == [
        "id",
        "after",
    ], f"answers.columns.tolist() == {answers.columns.tolist()}"
    assert len(new_test) + len(new_train) == len(
        old_train
    ), f"New train and test sets do not sum to old train set, got {len(new_test) + len(new_train)} and {len(old_train)}"

    # Write CSVs
    answers.to_csv(
        private / "answers.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC
    )
    sample_submission.to_csv(
        private / "sample_submission.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC
    )
    new_train.to_csv(
        public / "en_train.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC
    )
    new_test.to_csv(
        public / "en_test_2.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC
    )
    sample_submission.to_csv(
        public / "en_sample_submission_2.csv",
        index=False,
        quotechar='"',
        quoting=csv.QUOTE_NONNUMERIC,
    )

    # Zip up
    # with zipfile.ZipFile(public / "en_train.csv.zip", "w") as zipf:
    #     zipf.write(public / "en_train.csv", arcname="en_train.csv")
    # with zipfile.ZipFile(public / "en_test_2.csv.zip", "w") as zipf:
    #     zipf.write(public / "en_test_2.csv", arcname="en_test_2.csv")
    # with zipfile.ZipFile(public / "en_sample_submission_2.csv.zip", "w") as zipf:
    #     zipf.write(public / "en_sample_submission_2.csv", arcname="en_sample_submission_2.csv")
    # (public / "en_train.csv").unlink()
    # (public / "en_test_2.csv").unlink()
    # (public / "en_sample_submission_2.csv").unlink()

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving complete sentences (all tokens of a sentence must be included)
    
    Each row is a token with id format: sentence_id_token_id
    We sample complete sentences only, ensuring no partial sentences
    
    Only process test data in private_lite_dir, not touching public/train data
    
    Args:
        raw: Path to raw data (not used)
        lite_private: Path to private directory of prepared_lite 
        private: Path to private directory of prepared original
        max_test_samples: Maximum number of test samples (tokens/rows)
    """
    import pandas as pd
    import numpy as np
    
    print(f"Creating lite version with max {max_test_samples} test samples...")
    
    # Read test data from prepared/private
    answers_df = read_csv(private / "answers.csv")  # test with normalized text
    
    print(f"Original test samples (tokens): {len(answers_df)}")
    
    # Extract sentence_id from id column (format: sentence_id_token_id)
    answers_df['sentence_id'] = answers_df['id'].str.split('_').str[0].astype(int)
    answers_df['token_id'] = answers_df['id'].str.split('_').str[1].astype(int)
    
    # Group by sentence to understand sentence structure
    sentence_groups = answers_df.groupby('sentence_id').size().reset_index(name='token_count')
    sentence_groups = sentence_groups.sort_values('sentence_id')
    
    print(f"Original sentences: {len(sentence_groups)}")
    print(f"Average tokens per sentence: {sentence_groups['token_count'].mean():.1f}")
    print(f"Token count range: {sentence_groups['token_count'].min()}-{sentence_groups['token_count'].max()}")
    
    # If test set is already <= max_test_samples, keep original
    if len(answers_df) <= max_test_samples:
        print(f"Test set already has {len(answers_df)} tokens (<= {max_test_samples}), keeping original")
        # Copy only private directory
        import shutil
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # Select complete sentences that fit within max_test_samples
    print("Selecting complete sentences to maximize tokens within max_test_samples...")
    
    # Sort sentences by token count (ascending) for better packing
    sentence_groups_sorted = sentence_groups.sort_values('token_count').reset_index(drop=True)
    
    # Greedy selection to maximize total tokens
    selected_sentences = []
    total_tokens = 0
    
    # First pass: Add sentences greedily
    np.random.seed(42)
    shuffled_sentences = sentence_groups_sorted.sample(frac=1, random_state=42).reset_index(drop=True)
    
    for _, row in shuffled_sentences.iterrows():
        sentence_id = row['sentence_id']
        token_count = row['token_count']
        
        # Check if adding this sentence would exceed max_test_samples
        if total_tokens + token_count <= max_test_samples:
            selected_sentences.append(sentence_id)
            total_tokens += token_count
    
    # Second pass: Try to fill remaining space with smaller sentences
    remaining_space = max_test_samples - total_tokens
    if remaining_space > 0:
        print(f"Remaining space: {remaining_space} tokens, trying to fill with smaller sentences...")
        
        # Look for sentences that fit in remaining space
        unused_sentences = sentence_groups[~sentence_groups['sentence_id'].isin(selected_sentences)]
        candidates = unused_sentences[unused_sentences['token_count'] <= remaining_space]
        
        if len(candidates) > 0:
            # Sort by token count descending to get the largest that fits
            candidates = candidates.sort_values('token_count', ascending=False)
            
            for _, row in candidates.iterrows():
                sentence_id = row['sentence_id']
                token_count = row['token_count']
                
                if total_tokens + token_count <= max_test_samples:
                    selected_sentences.append(sentence_id)
                    total_tokens += token_count
                    remaining_space = max_test_samples - total_tokens
                    
                    if remaining_space == 0:
                        break
    
    print(f"Selected {len(selected_sentences)} complete sentences")
    print(f"Total tokens: {total_tokens} (≤ {max_test_samples})")
    
    # Filter answers to only include selected sentences
    sampled_test = answers_df[answers_df['sentence_id'].isin(selected_sentences)].copy()
    
    # Remove helper columns
    sampled_test = sampled_test.drop(columns=['sentence_id', 'token_id'])
    
    # Sort by id to maintain order
    sampled_test = sampled_test.sort_values('id').reset_index(drop=True)
    
    # Log final statistics
    final_sentence_ids = sampled_test['id'].str.split('_').str[0].astype(int).unique()
    print(f"Final statistics:")
    print(f"  Sentences: {len(final_sentence_ids)}")
    print(f"  Tokens: {len(sampled_test)}")
    print(f"  Sentence ID range: {min(final_sentence_ids)}-{max(final_sentence_ids)}")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_test.to_csv(lite_private / "answers.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    
    # Validation
    print("Running validation...")
    assert len(sampled_test) <= max_test_samples, f"Test set too large: {len(sampled_test)}"
    assert set(sampled_test.columns) == {"id", "after"}, "Should have id and after columns"
    
    # Verify all sentences are complete (no partial sentences)
    for sentence_id in final_sentence_ids:
        sentence_tokens = sampled_test[sampled_test['id'].str.startswith(f"{sentence_id}_")]
        token_ids = sentence_tokens['id'].str.split('_').str[1].astype(int).sort_values()
        expected_token_ids = list(range(len(token_ids)))
        actual_token_ids = token_ids.tolist()
        assert actual_token_ids == expected_token_ids, f"Sentence {sentence_id} has incomplete tokens: {actual_token_ids}"
    
    print(f"Successfully created lite version with {len(sampled_test)} tokens from {len(final_sentence_ids)} complete sentences")
    print(f"All sentences are complete (no partial sentences)")
    print(f"Private lite saved to: {lite_private}")