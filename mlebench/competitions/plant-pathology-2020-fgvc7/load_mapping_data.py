import os
import pandas as pd

def load_train_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    train_label_path = os.path.join(public_path, "train.csv")
    train_label_df = pd.read_csv(train_label_path)
    label_lookup = train_label_df.set_index("image_id").to_dict(orient="index")
    train_path = os.path.join(public_path, "images")
    data = []
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
    for file in os.listdir(train_path):
        if file.endswith(tuple(image_extensions)) and "Train" in file:
            file_name = file.split(".")[0]
            input_data = {
                "image_paths": [os.path.join(train_path, file)],
                "label": {
                    "healthy": label_lookup[file_name]["healthy"],
                    "multiple_diseases": label_lookup[file_name]["multiple_diseases"],
                    "rust": label_lookup[file_name]["rust"],
                    "scab": label_lookup[file_name]["scab"]
                },
                "input_id": file_name
            }
            data.append(input_data)
    return data, pd.DataFrame()


def load_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    private_path = os.path.join(path, "prepared", "private")
    test_path = os.path.join(public_path, "images")
    data = []
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
    for file in os.listdir(test_path):
        if file.endswith(tuple(image_extensions)) and "Test" in file:
            file_name = file.split(".")[0]
            input_data = {
                "image_paths": [os.path.join(test_path, file)],
                "input_id": file_name,
                "fields": {
                    "image_id": file_name
                },
                "predict_keys": ["prediction"]
            }
            data.append(input_data)
    return data, pd.DataFrame()

def load_lite_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    private_lite_path = os.path.join(path, "prepared_lite", "private")
    test_label_path = os.path.join(private_lite_path, "answers.csv")
    test_label_df = pd.read_csv(test_label_path)
    test_path = os.path.join(public_path, "images")
    data = []
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
    for file in os.listdir(test_path):
        if file.endswith(tuple(image_extensions)) and "Test" in file:
            file_name = file.split(".")[0]
            # Only include files that are in the lite test set
            if file_name in test_label_df["image_id"].values:
                input_data = {
                    "image_paths": [os.path.join(test_path, file)],
                    "input_id": file_name,
                    "fields": {
                        "image_id": file_name
                    },
                    "predict_keys": ["prediction"]
                }
                data.append(input_data)
    return data, test_label_df

def mapping(predictions: pd.DataFrame, path: str):
    submissions = pd.DataFrame({
        "image_id": predictions["image_id"],
        "healthy": (predictions["prediction"] == "healthy").astype(int),
        "multiple_diseases": (predictions["prediction"] == "multiple_diseases").astype(int),
        "rust": (predictions["prediction"] == "rust").astype(int),
        "scab": (predictions["prediction"] == "scab").astype(int)
    })
    return submissions
   