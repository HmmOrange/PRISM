import os
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

def create_sentence_map(df):
    return (
        df.groupby("sentence_id")["before"]
        .apply(lambda toks: [str(tok) for tok in toks if pd.notna(tok)])
        .to_dict()
    )

def build_train_row(row, sentence_map):
    sentence_id = row["sentence_id"]
    token_id = row["token_id"]
    return {
        "text_data": {
            "token": row["before"],
            "token_id": token_id,
            "sentence": " ".join(sentence_map.get(sentence_id, [])),
        },
        "label": {
            "nomalized_token": row["after"]
        },
        "input_id": f"{sentence_id}_{token_id}",
    }

def build_test_row(row, sentence_map):
    sentence_id = row["sentence_id"]
    token_id = row["token_id"]
    return {
        "text_data": {
            "token": row["before"],
            "token_id": token_id,
            "sentence": " ".join(sentence_map.get(sentence_id, [])),
        },
        "input_id": f"{sentence_id}_{token_id}",
        "fields": {
            "id": f"{sentence_id}_{token_id}"
        },
        "predict_keys": ["after"]
    }

def load_train_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    train_csv_path = os.path.join(public_path, "en_train.csv")
    train_df = pd.read_csv(train_csv_path)

    sentence_map = create_sentence_map(train_df)
    data = train_df.progress_apply(lambda row: build_train_row(row, sentence_map), axis=1).tolist()
    return data, pd.DataFrame()

def load_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    test_csv_path = os.path.join(public_path, "en_test_2.csv")
    test_df = pd.read_csv(test_csv_path)

    sentence_map = create_sentence_map(test_df)
    data = test_df.progress_apply(lambda row: build_test_row(row, sentence_map), axis=1).tolist()
    return data, pd.DataFrame()

def load_lite_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    private_lite_path = os.path.join(path, "prepared_lite", "private")
    test_csv_path = os.path.join(public_path, "en_test_2.csv")
    test_label_path = os.path.join(private_lite_path, "answers.csv")
    
    test_df = pd.read_csv(test_csv_path)
    test_label_df = pd.read_csv(test_label_path)
    
    # Extract sentence_ids from lite test labels (format: sentence_id_token_id)
    lite_sentence_ids = set(test_label_df["id"].str.split('_').str[0].astype(int))
    
    # Filter test data to only include sentences that are in the lite test set
    filtered_test_df = test_df[test_df["sentence_id"].isin(lite_sentence_ids)]
    
    sentence_map = create_sentence_map(filtered_test_df)
    data = filtered_test_df.progress_apply(lambda row: build_test_row(row, sentence_map), axis=1).tolist()
    
    return data, test_label_df

def mapping(predictions: pd.DataFrame, path: str):
    return predictions
