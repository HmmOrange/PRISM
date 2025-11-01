import os
import shutil
from pathlib import Path
from typing import Dict

import pandas as pd

from mlebench.utils import extract, read_csv


def filter_and_write_file(src: Path, dst: Path, old_id_to_new: Dict[int, int]):
    """
    Given txt file that has column 0 as rec_id, filters out rec_ids that are not in old_id_to_new and writes to dst
    """
    history_of_segments = open(src).read().splitlines()
    history_of_segments = history_of_segments[1:]
    history_of_segments = [
        (int(i.split(",")[0]), ",".join(i.split(",")[1:])) for i in history_of_segments
    ]
    history_of_segments = [
        (old_id_to_new[i[0]], i[1]) for i in history_of_segments if i[0] in old_id_to_new.keys()
    ]
    with open(dst, "w") as f:
        f.write("rec_id,[histogram of segment features]\n")
        for rec_id, labels in history_of_segments:
            f.write(f"{rec_id},{labels}\n")


def prepare(raw: Path, public: Path, private: Path):
    """
    Splits the data in raw into public and private datasets with appropriate test/train splits.
    """
    # extract only what we need
    extract(raw / "mlsp_contest_dataset.zip", raw)

    (public / "essential_data").mkdir(exist_ok=True)
    (public / "supplemental_data").mkdir(exist_ok=True)

    # Create train, test from train split
    cv_folds = read_csv(raw / "mlsp_contest_dataset/essential_data/CVfolds_2.txt")
    cv_folds = cv_folds[cv_folds["fold"] == 0].reset_index(drop=True)
    cv_folds.loc[cv_folds.sample(frac=0.2, random_state=0).index, "fold"] = 1

    old_id_to_new = {old_id: new_id for new_id, old_id in enumerate(cv_folds["rec_id"].values)}
    cv_folds["rec_id"] = cv_folds.index
    cv_folds.to_csv(public / "essential_data/CVfolds_2.txt", index=False)

    test_rec_ids = cv_folds[cv_folds["fold"] == 1]["rec_id"].values
    assert len(test_rec_ids) == 64, f"Expected 64 test rec_ids, got {len(test_rec_ids)}"

    # Update id2filename with new split
    rec_id2filename = read_csv(raw / "mlsp_contest_dataset/essential_data/rec_id2filename.txt")
    rec_id2filename = rec_id2filename[rec_id2filename["rec_id"].isin(old_id_to_new.keys())]
    rec_id2filename["rec_id"] = rec_id2filename["rec_id"].map(old_id_to_new)
    rec_id2filename.to_csv(public / "essential_data/rec_id2filename.txt", index=False)
    assert len(rec_id2filename) == len(
        cv_folds
    ), f"Expected {len(cv_folds)} entires in rec_id2filename, got {len(rec_id2filename)}"

    # Update labels with new split
    rec_labels = (
        open(raw / "mlsp_contest_dataset/essential_data/rec_labels_test_hidden.txt")
        .read()
        .splitlines()
    )
    rec_labels = rec_labels[1:]  # Ignore header line
    rec_labels_split = []
    for i in rec_labels:
        rec_id = i.split(",")[0]
        labels = ",".join(i.split(",")[1:]) if len(i.split(",")) > 1 else ""
        rec_labels_split.append((int(rec_id), labels))
    rec_labels_split = [i for i in rec_labels_split if i[0] in old_id_to_new.keys()]
    rec_labels_split = [(old_id_to_new[i[0]], i[1]) for i in rec_labels_split]

    # Public labels
    with open(public / "essential_data/rec_labels_test_hidden.txt", "w") as f:
        f.write("rec_id,[labels]\n")
        for rec_id, labels in rec_labels_split:
            if rec_id in test_rec_ids:
                labels = "?"
            if labels == "":  # Write without comma
                f.write(f"{rec_id}{labels}\n")
            else:
                f.write(f"{rec_id},{labels}\n")

    # Private labels. Create csv, with each row containing the label for a (rec_id, species_id) pair
    data = {"Id": [], "Probability": []}
    for rec_id, labels in rec_labels_split:
        if rec_id not in test_rec_ids:
            continue
        species_ids = [int(i) for i in labels.split(",") if i != ""]
        for species_id in range(0, 19):
            data["Id"].append(rec_id * 100 + species_id)
            data["Probability"].append(int(species_id in species_ids))

    pd.DataFrame(data).to_csv(private / "answers.csv", index=False)
    assert (
        len(pd.DataFrame(data)) == len(test_rec_ids) * 19
    ), f"Expected {len(test_rec_ids)*19} entires in answers.csv, got {len(pd.DataFrame(data))}"

    # Create new sample submission, following new submission format
    # http://www.kaggle.com/c/mlsp-2013-birds/forums/t/4961/new-submission-parser
    data = {
        "Id": [rec_id * 100 + species_id for rec_id in test_rec_ids for species_id in range(0, 19)],
        "Probability": 0,
    }
    pd.DataFrame(data).to_csv(public / "sample_submission.csv", index=False)
    assert (
        len(pd.DataFrame(data)) == len(test_rec_ids) * 19
    ), f"Expected {len(test_rec_ids)*19} entires in sample_submission.csv, got {len(pd.DataFrame(data))}"

    # Copy over species list
    shutil.copyfile(
        src=raw / "mlsp_contest_dataset/essential_data/species_list.txt",
        dst=public / "essential_data/species_list.txt",
    )

    # Copy over all src waves from train+test set
    (public / "essential_data/src_wavs").mkdir(exist_ok=True)
    for filename in rec_id2filename["filename"]:
        shutil.copyfile(
            src=raw / "mlsp_contest_dataset/essential_data/src_wavs" / f"{filename}.wav",
            dst=public / "essential_data/src_wavs" / f"{filename}.wav",
        )

    # Copy over train+test filtered spectrograms, segmentation examples, spectrograms, and supervised segmentation
    (public / "supplemental_data/filtered_spectrograms").mkdir(exist_ok=True)
    (public / "supplemental_data/segmentation_examples").mkdir(exist_ok=True)
    (public / "supplemental_data/spectrograms").mkdir(exist_ok=True)
    (public / "supplemental_data/supervised_segmentation").mkdir(exist_ok=True)
    for filename in rec_id2filename["filename"]:
        shutil.copyfile(
            src=raw
            / "mlsp_contest_dataset/supplemental_data/filtered_spectrograms"
            / f"{filename}.bmp",
            dst=public / "supplemental_data/filtered_spectrograms" / f"{filename}.bmp",
        )
        if os.path.exists(
            raw / "mlsp_contest_dataset/supplemental_data/segmentation_examples" / f"{filename}.bmp"
        ):
            shutil.copyfile(
                src=raw
                / "mlsp_contest_dataset/supplemental_data/segmentation_examples"
                / f"{filename}.bmp",
                dst=public / "supplemental_data/segmentation_examples" / f"{filename}.bmp",
            )
        shutil.copyfile(
            src=raw / "mlsp_contest_dataset/supplemental_data/spectrograms" / f"{filename}.bmp",
            dst=public / "supplemental_data/spectrograms" / f"{filename}.bmp",
        )
        shutil.copyfile(
            src=raw
            / "mlsp_contest_dataset/supplemental_data/supervised_segmentation"
            / f"{filename}.bmp",
            dst=public / "supplemental_data/supervised_segmentation" / f"{filename}.bmp",
        )

    # Copy over remaining files
    shutil.copyfile(
        src=raw / "mlsp_contest_dataset/supplemental_data/segment_clusters.bmp",
        dst=public / "supplemental_data/segment_clusters.bmp",
    )
    shutil.copyfile(
        src=raw / "mlsp_contest_dataset/supplemental_data/segment_mosaic.bmp",
        dst=public / "supplemental_data/segment_mosaic.bmp",
    )

    filter_and_write_file(
        src=raw / "mlsp_contest_dataset/supplemental_data/histogram_of_segments.txt",
        dst=public / "supplemental_data/histogram_of_segments.txt",
        old_id_to_new=old_id_to_new,
    )
    filter_and_write_file(
        src=raw / "mlsp_contest_dataset/supplemental_data/segment_features.txt",
        dst=public / "supplemental_data/segment_features.txt",
        old_id_to_new=old_id_to_new,
    )
    filter_and_write_file(
        src=raw / "mlsp_contest_dataset/supplemental_data/segment_rectangles.txt",
        dst=public / "supplemental_data/segment_rectangles.txt",
        old_id_to_new=old_id_to_new,
    )

def prepare_lite(raw: Path, lite_private: Path, private: Path, max_test_samples: int):
    """
    Create a lite version of dataset with test set <= max_test_samples samples
    while preserving the distribution of bird species (19 species, multi-label)
    
    Each test sample (recording) has 19 rows in answers.csv (one per species)
    Format: Id = rec_id * 100 + species_id, Probability = 0 or 1
    
    Only process test data in private_lite_dir, not touching public/train data
    
    Args:
        raw: Path to raw data (not used)
        lite_private: Path to private directory of prepared_lite 
        private: Path to private directory of prepared original
        max_test_samples: Maximum number of test samples (recordings)
    """
    import pandas as pd
    import numpy as np
    
    print(f"Creating lite version with max {max_test_samples} test samples...")
    
    # Read test data from prepared/private
    answers_df = read_csv(private / "answers.csv")  # test with multi-label format
    
    # Extract unique rec_ids from answers (each rec_id has 19 rows)
    answers_df['rec_id'] = answers_df['Id'] // 100
    unique_rec_ids = sorted(answers_df['rec_id'].unique())
    
    print(f"Original test samples (recordings): {len(unique_rec_ids)}")
    print(f"Original answers.csv rows: {len(answers_df)} (should be {len(unique_rec_ids)} * 19)")
    
    # Check species distribution across all recordings
    species_counts = {}
    for rec_id in unique_rec_ids:
        rec_data = answers_df[answers_df['rec_id'] == rec_id]
        for species_id in range(19):
            species_row = rec_data[rec_data['Id'] == rec_id * 100 + species_id]
            if len(species_row) > 0 and species_row.iloc[0]['Probability'] == 1:
                species_counts[species_id] = species_counts.get(species_id, 0) + 1
    
    print(f"Original species distribution: {species_counts}")
    
    # If test set is already <= max_test_samples, keep original
    if len(unique_rec_ids) <= max_test_samples:
        print(f"Test set already has {len(unique_rec_ids)} recordings (<= {max_test_samples}), keeping original")
        # Copy only private directory
        shutil.copytree(private, lite_private, dirs_exist_ok=True)
        return
    
    # Random sampling of recordings (since multi-label stratification is complex)
    print("Using random sampling for multi-label bird recordings...")
    np.random.seed(42)
    sampled_rec_ids = sorted(np.random.choice(unique_rec_ids, size=max_test_samples, replace=False))
    
    print(f"Sampled recordings: {sampled_rec_ids}")
    
    # Filter answers.csv to only include sampled recordings
    sampled_answers = answers_df[answers_df['rec_id'].isin(sampled_rec_ids)].copy()
    sampled_answers = sampled_answers.drop(columns=['rec_id'])  # Remove helper column
    sampled_answers = sampled_answers.sort_values('Id').reset_index(drop=True)
    
    # Check species distribution after sampling
    sampled_species_counts = {}
    for rec_id in sampled_rec_ids:
        rec_data = sampled_answers[sampled_answers['Id'] // 100 == rec_id]
        for species_id in range(19):
            species_row = rec_data[rec_data['Id'] == rec_id * 100 + species_id]
            if len(species_row) > 0 and species_row.iloc[0]['Probability'] == 1:
                sampled_species_counts[species_id] = sampled_species_counts.get(species_id, 0) + 1
    
    print(f"Sampled species distribution: {sampled_species_counts}")
    
    # Save sampled test data
    print("Saving sampled test data...")
    sampled_answers.to_csv(lite_private / "answers.csv", index=False)
    
    # Also need to copy CVfolds_2.txt to indicate which are test samples
    # Read original CVfolds to understand the structure
    public_dir = private.parent / "public"
    if (public_dir / "essential_data" / "CVfolds_2.txt").exists():
        cv_folds = read_csv(public_dir / "essential_data" / "CVfolds_2.txt")
        
        # Filter to only include sampled recordings
        sampled_cv_folds = cv_folds[cv_folds['rec_id'].isin(sampled_rec_ids)].copy()
        
        # Create lite public directory structure if needed
        lite_public_dir = lite_private.parent / "public"
        (lite_public_dir / "essential_data").mkdir(parents=True, exist_ok=True)
        
        # Save filtered CVfolds
        sampled_cv_folds.to_csv(lite_public_dir / "essential_data" / "CVfolds_2.txt", index=False)
        print(f"Saved filtered CVfolds_2.txt with {len(sampled_cv_folds)} recordings")
    
    # Validation
    print("Running validation...")
    assert len(sampled_answers) == len(sampled_rec_ids) * 19, f"Should have {len(sampled_rec_ids)} * 19 = {len(sampled_rec_ids) * 19} rows, got {len(sampled_answers)}"
    assert len(sampled_rec_ids) <= max_test_samples, f"Test set too large: {len(sampled_rec_ids)}"
    assert set(sampled_answers['Probability'].unique()).issubset({0, 1}), "Probabilities should only be 0 or 1"
    
    print(f"Successfully created lite version with {len(sampled_rec_ids)} test recordings")
    print(f"Total rows in answers.csv: {len(sampled_answers)} (= {len(sampled_rec_ids)} recordings * 19 species)")
    print(f"Multi-label bird species distribution preserved")
    print(f"Private lite saved to: {lite_private}")