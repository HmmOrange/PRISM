import os
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

def build_train_row(row):
    text_data = row.drop(["fare_amount", "key"]).to_dict()
    return {
        "text_data": text_data,
        "label": {
            "fare_amount": float(row["fare_amount"])
        },
        "input_id": row["key"],
    }

def build_test_row(row):
    text_data = row.drop(["key"]).to_dict()
    return {
        "text_data": text_data,
        "input_id": row["key"],
        "fields": row.to_dict(),
        "predict_keys": ["fare_amount"],
    }

def load_train_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    train_csv_path = os.path.join(public_path, "train.csv")
    train_df = pd.read_csv(train_csv_path)
    
    data = train_df.progress_apply(lambda row: build_train_row(row), axis=1).tolist()
    return data, pd.DataFrame()

def load_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    test_csv_path = os.path.join(public_path, "test.csv")
    test_df = pd.read_csv(test_csv_path)
    
    data = test_df.progress_apply(lambda row: build_test_row(row), axis=1).tolist()
    return data, pd.DataFrame()

def load_lite_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    private_lite_path = os.path.join(path, "prepared_lite", "private")
    test_csv_path = os.path.join(public_path, "test.csv")
    test_label_path = os.path.join(private_lite_path, "answers.csv")
    
    test_df = pd.read_csv(test_csv_path)
    test_label_df = pd.read_csv(test_label_path)
    
    # Filter test data to only include samples that are in the lite test set
    lite_keys = set(test_label_df["key"].values)
    filtered_test_df = test_df[test_df["key"].isin(lite_keys)]
    
    data = filtered_test_df.progress_apply(lambda row: build_test_row(row), axis=1).tolist()
    return data, test_label_df

def mapping(predictions: pd.DataFrame, path: str):
    return predictions
