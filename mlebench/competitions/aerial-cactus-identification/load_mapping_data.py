import os
import pandas as pd

def load_train_data(path: str):
   public_path = os.path.join(path, "prepared", "public")
   train_label_path = os.path.join(public_path, "train.csv")
   train_label_df = pd.read_csv(train_label_path)
   label_lookup = train_label_df.set_index("id").to_dict(orient="index")
   train_path = os.path.join(public_path, "train")
   data = []
   image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
   for file in os.listdir(train_path):
      input_data = {
         "image_paths": [os.path.join(train_path, file)],
         "input_id": file,
         "label": {
            "has_cactus": label_lookup[file]["has_cactus"]
         }
      }
      data.append(input_data)
   return data, train_label_df


def load_test_data(path: str):
   public_path = os.path.join(path, "prepared", "public")
   private_path = os.path.join(path, "prepared", "private")
   test_label_path = os.path.join(private_path, "answers.csv")
   test_label_df = pd.read_csv(test_label_path)
   test_path = os.path.join(public_path, "test")
   data = []
   image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
   for file in os.listdir(test_path):
      input_data = {
         "image_paths": [os.path.join(test_path, file)],
         "input_id": file,
         "fields": {
            "id": file
         },
         "predict_keys": ["has_cactus"]
      }
      data.append(input_data)
   return data, test_label_df

def load_lite_test_data(path: str):
   public_path = os.path.join(path, "prepared", "public")
   private_lite_path = os.path.join(path, "prepared_lite", "private")
   test_label_path = os.path.join(private_lite_path, "answers.csv")
   test_label_df = pd.read_csv(test_label_path)
   test_path = os.path.join(public_path, "test")
   data = []
   image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"]
   for file in os.listdir(test_path):
      # Only include files that are in the lite test set
      if file in test_label_df["id"].values:
         input_data = {
            "image_paths": [os.path.join(test_path, file)],
            "input_id": file,
            "fields": {
               "id": file
            },
            "predict_keys": ["has_cactus"]
         }
         data.append(input_data)
   return data, test_label_df

def mapping(predictions: pd.DataFrame, path: str) -> pd.DataFrame:
   return predictions