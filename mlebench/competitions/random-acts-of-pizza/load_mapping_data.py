import os
import json
import pandas as pd

def load_train_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    train_path = os.path.join(public_path, "train.json")
    with open(train_path, "r", encoding="utf-8") as file:
        train_data = json.loads(file.read())
    data = []
    for idx, item in enumerate(train_data):
        text_data = {key: item[key] for key in item.keys() if key != "requester_received_pizza"}
        input_data = {
            "text_data": {
                "text": item["request_text_edit_aware"]
            },
            "label": {
                "requester_received_pizza": item["requester_received_pizza"]
            },
            "input_id": item["request_id"],
        }
        data.append(input_data)
    return data, pd.DataFrame()

def load_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    test_path = os.path.join(public_path, "test.json")
    with open(test_path, "r", encoding="utf-8") as file:
        test_data = json.loads(file.read())
    data = []
    for idx, item in enumerate(test_data):
        input_data = {
            "text_data": {
                "text": item["request_text_edit_aware"]
            },
            "input_id": item["request_id"],
            "fields": {
                "request_id": item["request_id"]
            },
            "predict_keys": ["requester_received_pizza"]
        }
        data.append(input_data)
    return data, pd.DataFrame()

def load_lite_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    private_lite_path = os.path.join(path, "prepared_lite", "private")
    test_path = os.path.join(public_path, "test.json")
    test_label_path = os.path.join(private_lite_path, "answers.csv")
    
    # Load lite test labels to get filtered request IDs
    test_label_df = pd.read_csv(test_label_path)
    lite_request_ids = set(test_label_df["request_id"].values)
    
    # Load full test JSON data
    with open(test_path, "r", encoding="utf-8") as file:
        test_data = json.loads(file.read())
    
    # Filter test data to only include requests in lite set
    filtered_test_data = [item for item in test_data if item["request_id"] in lite_request_ids]
    
    data = []
    for idx, item in enumerate(filtered_test_data):
        input_data = {
            "text_data": {
                "text": item["request_text_edit_aware"]
            },
            "input_id": item["request_id"],
            "fields": {
                "request_id": item["request_id"]
            },
            "predict_keys": ["requester_received_pizza"]
        }
        data.append(input_data)
    
    return data, test_label_df

def mapping(predictions: pd.DataFrame, path: str):
   return predictions
