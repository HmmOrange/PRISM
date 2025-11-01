from multiprocessing import Value
import random
import os
import uuid
import base64
import json
from io import BytesIO
import re
import requests
import numpy as np
import torch
from PIL import Image, ImageDraw
from huggingface_hub import InferenceClient
from utils.constants import (
    config,
    HUGGINGFACE_HEADERS,
    PROXY,
    LOCAL_INFERENCE_ENDPOINT_URL,
    ROOT_PATH
)
from utils.common import image_to_bytes, audio_to_bytes, video_to_bytes
import soundfile as sf

model_desc = {}

with open(os.path.join(ROOT_PATH, "data", "huggingface_models.jsonl"), "r", encoding="utf-8") as f:
    for line in f.readlines():
        info = json.loads(line)
        model_desc[info["id"]] = {
            "metadata": info["metadata"] if "metadata" in info else {},
        }
        
FORBIDDEN_MODELS = ["facebook/bart-large-cnn"] ## List of models made model server crash @@
        
def check_data(task: str, data: dict):
    if task in ["token-classification", "text-generation", "translation", "summarization", "text-generation"] and "text" not in data:
        raise ValueError(f"Task {task} requires 'text' in data. Input data should be {{'text': str}}")
    if task in ["zero-shot-classification"] and ("text" not in data or "labels" not in data):
        raise ValueError(f"Task {task} requires 'text' and 'labels' in data. Input data should be {{'text': str, 'labels': list[str]}}")
    if task in ["question-answering"] and ("text" not in data or "context" not in data):
        raise ValueError(f"Task {task} requires 'text' and 'context' in data. Input data should be {{'text': str, 'context': str}}")
    if task in ["sentence-similarity"] and ("text" not in data or "other_sentences" not in data):
        raise ValueError(f"Task {task} requires 'text' and 'other_sentences' in data. Input data should be {{'text': str, 'other_sentences': list[str]}}")
    if task in ["tabular-classification", "tabular-regression"] and "row" not in data:
        raise ValueError(f"Task {task} requires 'row' in data. Input data should be {{'row': dict}}")
    if task in ["object-detection", "image-classification", "image-to-text"] and "image" not in data:
        raise ValueError(f"Task {task} requires 'image' in data. Input data should be {{'image': Image.Image}}")
    if task in ["automatic-speech-recognition", "audio-classification"] and "audio" not in data:
        raise ValueError(f"Task {task} requires 'audio' in data. Input data should be {{'audio': BytesIO}}")
    if task in ["video-classification"] and "video" not in data:
        raise ValueError(f"Task {task} requires 'video' in data. Input data should be {{'video': BytesIO}}   ")


def huggingface_model_inference(model_id: str, data: dict, task: str, return_resource: bool = False) -> dict:
    task_url = f"https://api-inference.huggingface.co/models/{model_id}"  # InferenceApi does not yet support some tasks

    client = InferenceClient(
        provider="hf-inference", api_key=config["huggingface"]["token"]
    )
    # print(task, model_id)
    if task in ["image-classification", "image-to-text"]:
        HUGGINGFACE_HEADERS["Content-Type"] = "image/jpeg"
        if isinstance(data["image"], str):
            base64_pattern = r"^[A-Za-z0-9+/=]+$"
            img_url_or_base64 = data["image"]
            if re.match(base64_pattern, img_url_or_base64):
                # If input is base64, decode to bytes
                img_data = base64.b64decode(img_url_or_base64)
            else:
                # If input is URL, download bytes from URL
                img_data = image_to_bytes(img_url_or_base64)
        elif isinstance(data["image"], Image.Image):
            # If input is PIL Image, save to bytes
            img_obj = data["image"]
            img_data = BytesIO()
            img_obj.save(img_data, format="PNG")
            img_data = img_data.getvalue()
        elif isinstance(data["image"], (np.ndarray, torch.Tensor, list)):
            raise ValueError(f"Input image must be a path, PIL Image, or bytes. Got {type(data['image'])}")
        else:
            # If input is bytes
            img_data = data["image"]
        r = requests.post(
            task_url, headers=HUGGINGFACE_HEADERS, data=img_data, proxies=PROXY
        )
        result = {}
        if task == "image-classification":
            # print(r)
            result["predicted"] = r.json()[0].pop("label")
        elif task == "image-to-text":
            result["text"] = r.json()[0].pop("generated_text")
        return result

    if task == "zero-shot-classification":
        text = data["text"]
        labels = data["labels"]
        r = requests.post(
            task_url,
            json={"inputs": text, "parameters": {"candidate_labels": labels}},
            headers=HUGGINGFACE_HEADERS,
        )
        result = {}
        result["predicted"] = r.json()["labels"][0]
        return result

    if task == "question-answering":
        result = client.question_answering(
            question=data["text"],
            context=data["context"] if "context" in data else "",
            model=model_id,
        )
        return result

    if task == "sentence-similarity":
        r = client.sentence_similarity(
            sentence=data["text"],
            other_sentences=(
                data["other_sentences"] if "other_sentences" in data else []
            ),
            model=model_id,
        )
        result = {"predicted": r}
        return result

    if task in ["translation"]:
        inputs = data["text"]
        result = client.translation(text=inputs, model=model_id)
        return result

    if task in ["summarization"]:
        if "file" in data:
            file = data["file"]
            with open(file, "r", encoding="utf-8") as f:
                inputs = f.read()
        else:
            inputs = data["text"]
        result = client.summarization(text=inputs, model=model_id)
        return result

    if task in [
        "text-classification",
        "text-generation",
    ]:
        inputs = data["text"]
        r = client.text_classification(text=inputs, model=model_id)
        result = {}
        if task == "text-classification":
            result["predicted"] = r[0].pop("label")
            if model_id == "mshenoda/roberta-spam":
                result["predicted"] = (
                    "spam" if result["predicted"] == "LABEL_1" else "ham"
                )
        if "id2label" in model_desc[model_id]["metadata"]:
            result["predicted"] = model_desc[model_id]["metadata"]["id2label"][result["predicted"]]
        return result

    if task == "token-classification":
        inputs = data["text"]
        r = client.token_classification(text=inputs, model=model_id)
        result = {"predicted": []}
        for item in r:
            result["predicted"].append(
                {
                    "word": item["word"],
                    "entity_group": item["entity_group"],
                }
            )
        return result

    if task in [
        "automatic-speech-recognition",
        "audio-classification",
    ]:
        audio = data["audio"]
        if isinstance(audio, str):
            base64_pattern = r"^[A-Za-z0-9+/=]+$"
            audio_url_or_base64 = audio
            if re.match(base64_pattern, audio_url_or_base64):
                audio_data = base64.b64decode(audio_url_or_base64)
            else:
                audio_data = audio_to_bytes(audio_url_or_base64)
        elif isinstance(audio, BytesIO):
            audio.seek(0)
            audio_data = audio.read()
        elif isinstance(audio, np.ndarray):
            audio_data = audio.tobytes()
        else:
            audio_data = audio
        client.headers["Content-Type"] = "audio/mpeg"
        if task == "automatic-speech-recognition":
            response = client.automatic_speech_recognition(audio=audio_data, model=model_id)
            result = response
        elif task == "audio-classification":
            response = client.audio_classification(audio=audio_data, model=model_id)
            result = response
        return result

    raise ValueError(f"Unsupported task: {task}")


def local_model_inference(model_id, data, task, return_resource: bool = False):
    task_url = f"{LOCAL_INFERENCE_ENDPOINT_URL}/models?model_id={model_id}"

    files = {}

    if task in ["image-classification", "object-detection", "image-to-text"]:
        # print(type(data["image"]))
        img_data = None
        # Handle different types of input
        if isinstance(data["image"], str):
            base64_pattern = r"^[A-Za-z0-9+/=]+$"
            img_url_or_base64 = data["image"]
            if re.match(base64_pattern, img_url_or_base64):
                # If input is base64, decode to bytes
                img_data = base64.b64decode(img_url_or_base64)
            else:
                # If input is URL, download bytes from URL
                img_data = image_to_bytes(img_url_or_base64)
        elif isinstance(data["image"], Image.Image):
            # If input is PIL Image, save to bytes
            img_obj = data["image"]
            img_data = BytesIO()
            img_obj.save(img_data, format="PNG")
            img_data = img_data.getvalue()
        elif isinstance(data["image"], (np.ndarray, torch.Tensor, list)):
            raise ValueError(f"Input image must be a path, PIL Image, or bytes. Got {type(data['image'])}")
        else:
            # If input is bytes
            img_data = data["image"]

        files["image"] = ("image.png", img_data, "image/png")
        data = {}

    elif task in [
        "automatic-speech-recognition",
        "audio-classification",
    ]:
        audio = data["audio"]
        if isinstance(audio, str):
            base64_pattern = r"^[A-Za-z0-9+/=]+$"
            audio_url_or_base64 = audio
            if re.match(base64_pattern, audio_url_or_base64):
                audio_data = base64.b64decode(audio_url_or_base64)
            else:
                audio_data = audio_to_bytes(audio_url_or_base64)
        else:
            audio_data = audio
        files["audio"] = ("audio.flac", audio_data, "audio/flac")
        data = {}

    response = requests.post(task_url, data={"data": json.dumps(data)}, files=files)
    # print(response.json())
    try:
        response = response.json()                         
        if response["code"] == "000":
            result = response["data"]
            if return_resource:
                if task == "object-detection":
                    image = Image.open(BytesIO(img_data))
                    draw = ImageDraw.Draw(image)
                    predictions = result["predicted"]
                    labels = list(item["label"] for item in predictions)
                    color_map = {}
                    for label in labels:
                        if label not in color_map:
                            color_map[label] = (random.randint(0, 255), random.randint(0, 100), random.randint(0, 255))
                    for prediction in predictions:
                        box = prediction["box"]
                        label = prediction["label"]
                        draw.rectangle(((box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])), outline=color_map[label], width=2)
                        draw.text((box["xmin"]+5, box["ymin"]-15), label, fill=color_map[label])
                    name = str(uuid.uuid4())[:4]
                    image.save(f"public/images/{name}.jpg")
                    result["generated image"] = f"public/images/{name}.jpg"
                if task == "image-to-text":
                    result["generated text"] = result["text"]
            return result
        else:
            raise ValueError(f"Error: {response['message']}")
    except Exception as e:
        raise ValueError(f"Error: {response}")

def model_inference(model_id: str, input_data: dict, hosted_on: str, task: str, return_resource: bool = False):
    if return_resource:
        os.makedirs("public/images", exist_ok=True)
        os.makedirs("public/audios", exist_ok=True)
        os.makedirs("public/videos", exist_ok=True)
    # if model_id not in model_desc:
    #     raise ValueError(f"Model {model_id} is not found")
    if model_id in ["your_model_id", "text-classification-model", "token-classification-model", "your_model_id_here", "whale_call_classifier"]:
        raise ValueError(f"Model {model_id} is not found")
    check_data(task, input_data)
    if model_id in FORBIDDEN_MODELS:
        raise ValueError(f"Model {model_id} is forbidden")
    try:
        if False or model_id in ["openai/whisper-large-v3"]:
            try:
                return huggingface_model_inference(model_id, input_data, task, return_resource)
            except Exception as e:
                return local_model_inference(model_id, input_data, task, return_resource)
        elif hosted_on == "local":
            return local_model_inference(model_id, input_data, task, return_resource)
        else:
            return local_model_inference(model_id, input_data, task, return_resource)
            # raise ValueError(f"Unsupported hosted_on: {hosted_on}")
    except Exception as e:
        raise e


if __name__ == "__main__":
    data=[{'image_paths': ['tasks/graph-level/dog_breed_or_cat_count/validation/inputs/1/image.jpg'], 'label': '2', 'input_id': '1'}]
    sample_data = data[0] 
    image_path = sample_data['image_paths'][0]
    image = Image.open(image_path)
    model_ids = ['facebook/detr-resnet-101']
    outputs = {}
    for model_id in model_ids:
       try:
           output = model_inference(model_id=model_id, input_data={'image': image}, hosted_on='local', task='object-detection', return_resource=True)
           outputs[model_id] = output
       except Exception as e:
           import traceback
           traceback.print_exc()
           outputs[model_id] = str(e)
    print(outputs)