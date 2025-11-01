import os
from typing import Optional
import re
import pandas as pd

def load_train_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    train_path = os.path.join(public_path, "train2")
    data = []
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".aif"]
    for file in os.listdir(train_path):
        if file.endswith(tuple(audio_extensions)):
            input_data = {
                "audio_paths": [os.path.join(train_path, file)],
                "input_id": file,
                "label": {
                    "probability": 1 if file.endswith("_1.aif") else 0
                }
            }
            data.append(input_data)
    return data, pd.DataFrame()


def load_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    test_path = os.path.join(public_path, "test2")
    data = []
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".aif"]
    for file in os.listdir(test_path):
        if file.endswith(tuple(audio_extensions)):
            input_data = {
                "audio_paths": [os.path.join(test_path, file)],
                "input_id": file,
                "fields": {
                    "clip": file
                },
                "predict_keys": ["probability"]
            }
            data.append(input_data)
    return data, pd.DataFrame()

def load_lite_test_data(path: str):
    public_path = os.path.join(path, "prepared", "public")
    private_lite_path = os.path.join(path, "prepared_lite", "private")
    test_path = os.path.join(public_path, "test2")
    test_label_path = os.path.join(private_lite_path, "answers.csv")
    
    test_label_df = pd.read_csv(test_label_path)
    
    # Get list of audio files that are in the lite test set
    lite_clips = set(test_label_df["clip"].values)
    
    data = []
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".aif"]
    for file in os.listdir(test_path):
        if file.endswith(tuple(audio_extensions)):
            # Only include files that are in the lite test set
            if file in lite_clips:
                input_data = {
                    "audio_paths": [os.path.join(test_path, file)],
                    "input_id": file,
                    "fields": {
                        "clip": file
                    },
                    "predict_keys": ["probability"]
                }
                data.append(input_data)
    
    return data, test_label_df

def mapping(predictions: pd.DataFrame, path: str):
    return predictions