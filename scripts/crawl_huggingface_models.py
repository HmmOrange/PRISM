import json
import click
from huggingface_hub import HfApi
from huggingface_hub import ModelCard
from tqdm import tqdm 

api = HfApi()

MAX_MODELS = 700
REQUIRED_TAG = "transformers" # For support transformers pipeline
MAX_PARAMETERS = 500_000_000
ALLOWED_PIPELINE_TAGS = [ # For model with mesurable output
    "token-classification",
    "text-classification",
    "zero-shot-classification",
    "translation",
    "summarization",
    "question-answering",
    "text-generation",
    "sentence-similarity",
    "tabular-classification",
    "object-detection",
    "image-classification",
    "image-to-text",
    "automatic-speech-recognition",
    "audio-classification",
    "tabular-regression",
    "video-classification",
] 

MAX_MODEL_PER_PIPELINE_TAG = MAX_MODELS // len(ALLOWED_PIPELINE_TAGS)

def get_model_parameters(model_info):
    """Estimate number of parameters from safetensors, config, or file size."""
    # Check safetensors
    if hasattr(model_info, 'safetensors') and model_info.safetensors:
        total_params = model_info.safetensors.get('total', None)
        if total_params is not None and total_params > 0:
            return total_params
    return float('inf')

def get_model_information(model_id: str, pipeline_tag: str, limit: bool = True) -> dict:
    model_info = api.model_info(model_id)
    tags = model_info.tags if hasattr(model_info, "tags") else []
    if limit and REQUIRED_TAG not in tags:
        return None
    downloads = model_info.downloads
    likes = model_info.likes
    inference = model_info.inference
    num_parameters = get_model_parameters(model_info)
    if limit and num_parameters and num_parameters > MAX_PARAMETERS:
        return None
    try:
        model_card = ModelCard.load(model_id)
        description = model_card.content
    except:
        if limit:
            return None
        else:
            description = ""
    print()
    print("Model: ", model_id)
    print("Parameters: ", num_parameters//1000000, "M")
    card_data = model_info.card_data
    meta = card_data.to_dict() if card_data else {}
    model_information = {
        "id": model_id,
        "pipeline_tag": pipeline_tag,
        "tags": tags,
        "description": description,
        "downloads": downloads,
        "likes": likes,
        "meta": meta,
        "inference_type": "huggingface" if inference == "warm" else "local"
    }
    return model_information

with open("data/model_desc.jsonl", "r", encoding="utf-8") as f:
    model_descs = [json.loads(line) for line in f]
    model_ids = [model_desc["id"] for model_desc in model_descs]

def crawl_models():
    models_data = []
    for model_desc in tqdm(model_descs, desc="Crawling models", ncols=80):
        model_id = model_desc["id"]
        model_information = get_model_information(model_id=model_id, pipeline_tag=model_desc["tag"], limit=False)
        if not model_information:
            continue
        model_information["inference_type"] = model_desc["inference_type"]
        models_data.append(model_information)
    print(len(models_data))

    for pipeline_tag in tqdm(ALLOWED_PIPELINE_TAGS, desc="Crawling pipeline tags", ncols=80):
        pipeline_models_data = []
        models = api.list_models(
            sort="downloads",
            direction=-1,
            cardData=True,
            full=True,
            pipeline_tag=pipeline_tag
        )
        pbar = tqdm(desc=f"Crawling {pipeline_tag}", total=MAX_MODEL_PER_PIPELINE_TAG, ncols=80)
        for model in models:
            if len(pipeline_models_data) > MAX_MODEL_PER_PIPELINE_TAG:
                break
            model_id = model.id
            if model_id in model_ids:
                continue
            model_information = get_model_information(model_id=model_id, pipeline_tag=pipeline_tag)
            if not model_information:
                continue
            pipeline_models_data.append(model_information)
            pbar.update(1)
        models_data.extend(pipeline_models_data)
        print(pipeline_tag, len(models_data), MAX_MODEL_PER_PIPELINE_TAG)
    with open("data/huggingface_models.jsonl", "w", encoding="utf-8") as f:
        for model in models_data:
            f.write(json.dumps(model, ensure_ascii=False) + "\n")
    
@click.command()
def main():
    models_data = crawl_models()
    
if __name__ == "__main__":
    main()