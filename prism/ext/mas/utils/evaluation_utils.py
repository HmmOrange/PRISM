import importlib
import importlib.util
import os
from pathlib import Path
from typing import Callable, Tuple, Any, Union
import pandas as pd
from copy import deepcopy

from scripts.calculator import Calculator, MetricType
from mlebench.grade import grade_csv
from mlebench.registry import registry

class EvaluationUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.calculator = Calculator()
        
    def dict_to_description(self, data: dict) -> str:
        parts = []

        # Image paths
        if "image_paths" in data and data["image_paths"]:
            images = ", ".join(data["image_paths"])
            parts.append(f"Image paths: {images}")

        # Video paths
        if "video_paths" in data and data["video_paths"]:
            videos = ", ".join(data["video_paths"])
            parts.append(f"Video paths: {videos}")

        # Audio paths
        if "audio_paths" in data and data["audio_paths"]:
            audios = ", ".join(data["audio_paths"])
            parts.append(f"Audio paths: {audios}")

        # Text data
        if "text_data" in data and data["text_data"]:
            text_fields = ", ".join([f"{k}: {v}" for k, v in data["text_data"].items()])
            parts.append(f"Text data has fields {text_fields}")

        return "\n".join(parts)
        
    async def evaluate_workflow(self, workflow_class: Callable, directory: str, data: pd.DataFrame, label: pd.DataFrame, metric: str) -> Tuple[float, list[str]]:
        workflow = workflow_class()
        metric_type = MetricType(metric)
        predictions = []
        logs = []
        for _, row in data.iterrows():
            input_id = row["input_id"]
            log_row_dict = deepcopy(row.to_dict())
            if "text_data" in log_row_dict and "text" in log_row_dict["text_data"]:
                log_row_dict["text_data"]["text"] = log_row_dict["text_data"]["text"][:200]
            input_description = self.dict_to_description(log_row_dict)
            try:
                prediction = await workflow(row.to_dict())
                
                logs.append( f"Input: {input_description}\nPrediction: {str(prediction)[:100] + '...' if len(str(prediction)) > 100 else str(prediction)}\n Expected Output: {str(row['label'])[:100] + '...' if len(str(row['label'])) > 100 else str(row['label'])}\n")
                predictions.append({"id": input_id, "prediction": prediction})
            except Exception as e:
                logs.append(f"Input: {input_description}\nError: {str(e)[:100] + '...' if len(str(e)) > 100 else str(e)}\n Expected Output: {str(row['label'])[:100] + '...' if len(str(row['label'])) > 100 else str(row['label'])}\n")
                predictions.append({"id": input_id, "prediction": ""})
        _, df_score = self.calculator(metric_type, pd.DataFrame(predictions), label)
        
        return df_score["score"].mean(), logs
 
    
    def format_competition_label(self, label: dict, competition_id: str) -> any:
        if competition_id == "aerial-cactus-identification":
            return label["cactus"]
        elif competition_id == "aptos2019-blindness-detection":
            return label["diagnosis"]
        elif competition_id == "detecting-insults-in-social-commentary":
            return label["Insult"]
        elif competition_id == "dog-breed-identification":
            return label["breed"]
        elif competition_id == "dogs-vs-cats-redux-kernels-edition":
            return 1 if label["animal"] == "dog" else 0
        elif competition_id == "histopathologic-cancer-detection":
            return label["label"]
        elif competition_id == "jigsaw-toxic-comment-classification-challenge":
            return [value for item, value in label.items()]
        elif competition_id == "leaf-classification":
            return label["species"]
        elif competition_id == "mlsp-2013-birds":
            return label["species"]
        elif competition_id == "new-york-city-taxi-fare-prediction":
            return label["fare_amount"]
        elif competition_id == "nomad2018-predict-transparent-conductors":
            return [value for item, value in label.items()]
        elif competition_id == "plant-pathology-2020-fgvc7":
            for item, value in label.items():
                if value == 1:
                    return item
        elif competition_id == "random-acts-of-pizza":
            return label["requester_received_pizza"]
        elif competition_id == "ranzcr-clip-catheter-line-classification":
            return [value for item, value in label.items()]
        elif competition_id == "siim-isic-melanoma-classification":
            return label["target"]
        elif competition_id == "spooky-author-identification":
            return label["author"]
        elif competition_id == "tabular-playground-series-dec-2021":
            return label["Cover_Type"]
        elif competition_id == "tabular-playground-series-may-2022":
            return label["target"]
        elif competition_id == "text-normalization-challenge-english-language":
            return label["nomalized_token"]
        elif competition_id == "text-normalization-challenge-russian-language":
            return label["nomalized_token"]
        elif competition_id == "the-icml-2013-whale-challenge-right-whale-redux":
            return label["probability"]
        else:
            raise ValueError(f"Competition ID {competition_id} not found")
        
    async def evaluate_mlebench_workflow(self, workflow_class: Callable, data: pd.DataFrame, competition_id: str) -> Tuple[float, list[str]]:
        workflow = workflow_class()
        logs = []
        trues = 0
        for _, row in data.iterrows():
            row_copy = deepcopy(row)
            del row_copy["label"]
            del row_copy["input_id"]
            label = row["label"]
            formatted_label = self.format_competition_label(label, competition_id)
            try:
                prediction = await workflow(row.to_dict())
                logs.append(f"Input: {row_copy.to_dict()}\nPrediction: {prediction}\n Expected Output: {formatted_label}\n")
                print(prediction, formatted_label)
                if prediction == formatted_label:
                    trues += 1
            except Exception as e:
                logs.append(f"Input: {row_copy.to_dict()}\nError: {str(e)[:100] + '...' if len(str(e)) > 100 else str(e)}\n Expected Output: {label}\n")
        print("Accuracy: ", trues / len(data))
        return trues / len(data), logs
        
    
    def log_results(self, directory: str):
        pass