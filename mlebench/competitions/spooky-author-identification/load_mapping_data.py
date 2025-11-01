import os
import pandas as pd

def load_train_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    train_csv_path = os.path.join(public_path, "train.csv")
    train_df = pd.read_csv(train_csv_path)
    data = []
    for idx, row in train_df.iterrows():
        input_data = {
            "text_data": {
                "text": row["text"]
            },
            "label": {
                "author": row["author"]
            },
            "input_id": row["id"],
        }
        data.append(input_data)
    return data, pd.DataFrame()

def load_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    test_csv_path = os.path.join(public_path, "test.csv")
    test_df = pd.read_csv(test_csv_path)
    data = []
    for idx, row in test_df.iterrows():
        input_data = {
            "text_data": {
                "text": row["text"]
            },
            "input_id": row["id"],
            "fields": {
                "id": row["id"]
            },
            "predict_keys": ["author"]
        }
        data.append(input_data)
    return data, pd.DataFrame()

def load_lite_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    private_lite_path = os.path.join(path, "prepared_lite", "private")
    test_csv_path = os.path.join(public_path, "test.csv")
    test_label_path = os.path.join(private_lite_path, "answers.csv")
    
    test_df = pd.read_csv(test_csv_path)
    test_label_df = pd.read_csv(test_label_path)
    
    # Filter test data to only include samples that are in the lite test set
    lite_ids = set(test_label_df["id"].values)
    filtered_test_df = test_df[test_df["id"].isin(lite_ids)]
    
    data = []
    for idx, row in filtered_test_df.iterrows():
        input_data = {
            "text_data": {
                "text": row["text"]
            },
            "input_id": row["id"],
            "fields": {
                "id": row["id"]
            },
            "predict_keys": ["author"]
        }
        data.append(input_data)
    
    return data, test_label_df

def mapping(predictions: pd.DataFrame, path: str):
    
    submissions = pd.DataFrame({
        "id": predictions["id"],
        "EAP": (predictions["author"] == "EAP").astype(int),
        "HPL": (predictions["author"] == "HPL").astype(int),
        "MWS": (predictions["author"] == "MWS").astype(int),
    })

    return submissions
