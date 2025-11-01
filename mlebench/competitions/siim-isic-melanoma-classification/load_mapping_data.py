import os
import pandas as pd

def load_train_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    train_label_path = os.path.join(public_path, "train.csv")
    train_label_df = pd.read_csv(train_label_path)
    label_lookup = train_label_df.set_index("image_name").to_dict(orient="index")
    train_path = os.path.join(public_path, "jpeg", "train")
    data = []
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
    for file in os.listdir(train_path):
        if file.endswith(tuple(image_extensions)):
            file_name = file.split(".jpg")[0]
            input_data = {
                "image_paths": [os.path.join(train_path, file)],
                "text_data": {
                    "patient_id": label_lookup[file_name]["patient_id"],
                    "sex": label_lookup[file_name]["sex"],
                    "age_approx": label_lookup[file_name]["age_approx"],
                    "anatom_site_general_challenge": label_lookup[file_name]["anatom_site_general_challenge"],
                },
                "label": {
                    "target": label_lookup[file_name]["target"]
                },
                "input_id": file_name
            }
            data.append(input_data)
    return data, pd.DataFrame()


def load_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    test_label_path = os.path.join(public_path, "test.csv")
    test_label_df = pd.read_csv(test_label_path)
    label_lookup = test_label_df.set_index("image_name").to_dict(orient="index")
    private_path = os.path.join(path, "prepared", "private")
    test_path = os.path.join(public_path, "jpeg", "test")
    data = []
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
    for file in os.listdir(test_path):
        if file.endswith(tuple(image_extensions)):
            file_name = file.split(".jpg")[0]
            input_data = {
                "image_paths": [os.path.join(test_path, file)],
                "text_data": {
                    "patient_id": label_lookup[file_name]["patient_id"],
                    "sex": label_lookup[file_name]["sex"],
                    "age_approx": label_lookup[file_name]["age_approx"],
                    "anatom_site_general_challenge": label_lookup[file_name]["anatom_site_general_challenge"],
                },
                "input_id": file_name,
                "fields": {
                    "image_name": file_name
                },
                "predict_keys": ["target"]
            }
            data.append(input_data)
    return data, pd.DataFrame()

def load_lite_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    private_lite_path = os.path.join(path, "prepared_lite", "private")
    test_label_path = os.path.join(private_lite_path, "answers.csv")
    test_label_df = pd.read_csv(test_label_path)
    
    # Load metadata from public test.csv for additional patient information
    public_test_path = os.path.join(public_path, "test.csv")
    public_test_df = pd.read_csv(public_test_path)
    metadata_lookup = public_test_df.set_index("image_name").to_dict(orient="index")
    
    test_path = os.path.join(public_path, "jpeg", "test")
    data = []
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
    for file in os.listdir(test_path):
        if file.endswith(tuple(image_extensions)):
            file_name = file.split(".jpg")[0]
            # Only include files that are in the lite test set
            if file_name in test_label_df["image_name"].values:
                # Get metadata if available
                metadata = metadata_lookup.get(file_name, {})
                input_data = {
                    "image_paths": [os.path.join(test_path, file)],
                    "text_data": {
                        "patient_id": metadata.get("patient_id", ""),
                        "sex": metadata.get("sex", ""),
                        "age_approx": metadata.get("age_approx", ""),
                        "anatom_site_general_challenge": metadata.get("anatom_site_general_challenge", ""),
                    },
                    "input_id": file_name,
                    "fields": {
                        "image_name": file_name
                    },
                    "predict_keys": ["target"]
                }
                data.append(input_data)
    return data, test_label_df

def mapping(predictions: pd.DataFrame, path: str):
   return predictions
   