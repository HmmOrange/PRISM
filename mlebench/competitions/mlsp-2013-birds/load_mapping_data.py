import os
import pandas as pd

def read_rec_labels(file_path: str) -> pd.DataFrame:
    data = []
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
        
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.strip().split(",")
        rec_id = int(parts[0])
        if len(parts) == 1:
            labels = []
        elif parts[1] == "?":
            labels = ["?"]
        else:
            labels = [int(x) for x in parts[1:] if x.strip().isdigit()]
        data.append({"rec_id": rec_id, "labels": ",".join([str(x) for x in labels])})
    return pd.DataFrame(data)

def load_train_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    audio_path = os.path.join(public_path, "essential_data", "src_wavs")
    rec_id2filename_path = os.path.join(public_path, "essential_data", "rec_id2filename.txt")
    rec_id2filename_df = pd.read_csv(rec_id2filename_path)
    rec_id2filename_df = rec_id2filename_df.set_index("rec_id")
    rec_labels_test_hidden_path = os.path.join(public_path, "essential_data", "rec_labels_test_hidden.txt") 
    rec_labels_test_hidden_df = read_rec_labels(rec_labels_test_hidden_path)
    rec_labels_test_hidden_df = rec_labels_test_hidden_df.set_index("rec_id")
    species_list_path = os.path.join(public_path, "essential_data", "species_list.txt")
    species_list = pd.read_csv(species_list_path)
    species_list = species_list.set_index("class_id")
    cvfolds_path = os.path.join(public_path, "essential_data", "cvfolds_2.txt")
    cvfolds_df = pd.read_csv(cvfolds_path)
    data = []
    for i, row in cvfolds_df.iterrows():
        fold = row["fold"]
        if int(fold) == 1:
            continue
        rec_id = row["rec_id"]
        label_str = rec_labels_test_hidden_df.loc[rec_id, "labels"]
        if label_str == "?":
            species_names = []
        else:
            class_ids = [int(x) for x in label_str.split(",") if x.strip().isdigit()]
            species_names = [species_list.loc[cid, "species"] for cid in class_ids]
        filename = rec_id2filename_df.loc[rec_id, "filename"]
        input_data = {
            "audio_paths": [os.path.join(audio_path, f"{filename}.wav")],
            "input_id": filename,
            "label": {
                "species": species_names
            }
        }
        data.append(input_data)
    return data, pd.DataFrame()
        
def load_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    audio_path = os.path.join(public_path, "essential_data", "src_wavs")
    rec_id2filename_path = os.path.join(public_path, "essential_data", "rec_id2filename.txt")
    rec_id2filename_df = pd.read_csv(rec_id2filename_path)
    rec_id2filename_df = rec_id2filename_df.set_index("rec_id")
    species_list_path = os.path.join(public_path, "essential_data", "species_list.txt")
    species_list = pd.read_csv(species_list_path)
    species_list = species_list.set_index("class_id")
    cvfolds_path = os.path.join(public_path, "essential_data", "cvfolds_2.txt")
    cvfolds_df = pd.read_csv(cvfolds_path)
    data = []
    for i, row in cvfolds_df.iterrows():
        fold = row["fold"]
        if int(fold) == 0:
            continue
        rec_id = row["rec_id"]
        filename = rec_id2filename_df.loc[rec_id, "filename"]
        input_data = {
            "audio_paths": [os.path.join(audio_path, f"{filename}.wav")],
            "input_id": filename,
            "fields": {
                "Id": rec_id
            },
            "predict_keys": ["prediction"]
        }
        data.append(input_data)
    return data, pd.DataFrame()

def load_lite_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    private_lite_path = os.path.join(path, "prepared_lite", "private")
    
    # Load lite CVfolds to get sampled recordings
    lite_public_path = os.path.join(path, "prepared_lite", "public")
    cvfolds_path = os.path.join(lite_public_path, "essential_data", "cvfolds_2.txt")
    cvfolds_df = pd.read_csv(cvfolds_path)
    
    # Get test recordings (fold == 1) from lite set
    test_recordings = cvfolds_df[cvfolds_df["fold"] == 1]["rec_id"].values
    
    # Load other necessary data from original public (these are shared)
    audio_path = os.path.join(public_path, "essential_data", "src_wavs")
    rec_id2filename_path = os.path.join(public_path, "essential_data", "rec_id2filename.txt")
    rec_id2filename_df = pd.read_csv(rec_id2filename_path)
    rec_id2filename_df = rec_id2filename_df.set_index("rec_id")
    
    species_list_path = os.path.join(public_path, "essential_data", "species_list.txt")
    species_list = pd.read_csv(species_list_path)
    species_list = species_list.set_index("class_id")
    
    # Load lite test labels
    test_label_path = os.path.join(private_lite_path, "answers.csv")
    test_label_df = pd.read_csv(test_label_path)
    
    data = []
    for rec_id in test_recordings:
        # Check if this recording exists in the filename mapping
        if rec_id in rec_id2filename_df.index:
            filename = rec_id2filename_df.loc[rec_id, "filename"]
            input_data = {
                "audio_paths": [os.path.join(audio_path, f"{filename}.wav")],
                "input_id": filename,
                "fields": {
                    "Id": rec_id
                },
                "predict_keys": ["prediction"]
            }
            data.append(input_data)
    
    return data, test_label_df

def mapping(predictions: pd.DataFrame, path: str):
    output_rows = []
    for _, row in predictions.iterrows():
        rec_id = int(row["Id"])
        predicted_species = row["prediction"]
        
        # Ensure list of ints (some predictions may be string if not preprocessed)
        if isinstance(predicted_species, str):
            predicted_species = [int(x) for x in predicted_species.split(",") if x.strip().isdigit()]
        
        for species_id in range(19):
            submission_id = rec_id * 100 + species_id
            prob = 1 if species_id in predicted_species else 0
            output_rows.append({
                "Id": submission_id,
                "Probability": prob
            })

    submissions = pd.DataFrame(output_rows).sort_values("Id")
    return submissions