import os
import pandas as pd

def load_train_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    train_path = os.path.join(public_path, "train")
    train_cleaned_path = os.path.join(public_path, "train_cleaned")
    data = []
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
    for file in os.listdir(train_path):
        if file.endswith(tuple(image_extensions)):
            input_data = {
                "image_paths": [os.path.join(train_path, file)],
                "label": {
                    "cleaned_image_path": [os.path.join(train_cleaned_path, file)]
                },
                "input_id": file
            }
            data.append(input_data)
    return data, pd.DataFrame()


def load_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    private_path = os.path.join(path, "prepared", "private")
    test_path = os.path.join(public_path, "test")
    data = []
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
    for file in os.listdir(test_path):
        if file.endswith(tuple(image_extensions)):
            file_name =file.split(".")[0]
            input_data = {
                "image_paths": [os.path.join(test_path, file)],
                "input_id": file_name,
                "fields": {
                    "id": file_name
                },
                "predict_keys": ["value"]
            }
            data.append(input_data)
    return data, pd.DataFrame()


def mapping(predictions: pd.DataFrame, path: str):
   return predictions
   