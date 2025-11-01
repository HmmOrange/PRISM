import os
import pandas as pd

def load_train_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    train_label_path = os.path.join(public_path, "train.csv")
    train_label_df = pd.read_csv(train_label_path)
    label_lookup = train_label_df.set_index("StudyInstanceUID").to_dict(orient="index")
    train_path = os.path.join(public_path, "train")
    data = []
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
    for file in os.listdir(train_path):
        if file.endswith(tuple(image_extensions)):
            file_name = file.split(".jpg")[0]
            input_data = {
                "image_paths": [os.path.join(train_path, file)],
                "label": {
                    "ETT - Abnormal": label_lookup[file_name]["ETT - Abnormal"],
                    "ETT - Borderline": label_lookup[file_name]["ETT - Borderline"],
                    "ETT - Normal": label_lookup[file_name]["ETT - Normal"],
                    "NGT - Abnormal": label_lookup[file_name]["NGT - Abnormal"],
                    "NGT - Borderline": label_lookup[file_name]["NGT - Borderline"],
                    "NGT - Incompletely Imaged": label_lookup[file_name]["NGT - Incompletely Imaged"],
                    "NGT - Normal": label_lookup[file_name]["NGT - Normal"],
                    "CVC - Abnormal": label_lookup[file_name]["CVC - Abnormal"],
                    "CVC - Borderline": label_lookup[file_name]["CVC - Borderline"],
                    "CVC - Normal": label_lookup[file_name]["CVC - Normal"],
                    "Swan Ganz Catheter Present": label_lookup[file_name]["Swan Ganz Catheter Present"]
                },
                "input_id": file_name
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
            file_name = file.split(".jpg")[0]
            input_data = {
                "image_paths": [os.path.join(test_path, file)],
                "input_id": file_name,
                "fields": {
                    "StudyInstanceUID": file_name
                },
                "predict_keys": ["ETT - Abnormal", "ETT - Borderline", "ETT - Normal", "NGT - Abnormal", "NGT - Borderline", "NGT - Incompletely Imaged", "NGT - Normal", "CVC - Abnormal", "CVC - Borderline", "CVC - Normal", "Swan Ganz Catheter Present"]
            }
            data.append(input_data)
    return data, pd.DataFrame()

def load_lite_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    private_lite_path = os.path.join(path, "prepared_lite", "private")
    test_label_path = os.path.join(private_lite_path, "answers.csv")
    test_label_df = pd.read_csv(test_label_path)
    test_path = os.path.join(public_path, "test")
    data = []
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
    for file in os.listdir(test_path):
        if file.endswith(tuple(image_extensions)):
            file_name = file.split(".jpg")[0]
            # Only include files that are in the lite test set
            if file_name in test_label_df["StudyInstanceUID"].values:
                input_data = {
                    "image_paths": [os.path.join(test_path, file)],
                    "input_id": file_name,
                    "fields": {
                        "StudyInstanceUID": file_name
                    },
                    "predict_keys": ["ETT - Abnormal", "ETT - Borderline", "ETT - Normal", "NGT - Abnormal", "NGT - Borderline", "NGT - Incompletely Imaged", "NGT - Normal", "CVC - Abnormal", "CVC - Borderline", "CVC - Normal", "Swan Ganz Catheter Present"]
                }
                data.append(input_data)
    return data, test_label_df

def mapping(predictions: pd.DataFrame, path: str):
   return predictions
   