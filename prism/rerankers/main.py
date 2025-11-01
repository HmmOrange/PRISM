import re
from pyexpat import model
import yaml
from openai import OpenAI
import numpy as np
from storage.model_storage import model_storage
import ast

config_path = "configs/config.yaml"

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

embedding_cfg = config["embedding"]
api_key = embedding_cfg["api_key"]
model_name = embedding_cfg["model"]

# Khởi tạo client OpenAI
client = OpenAI(api_key=api_key, base_url=embedding_cfg.get("base_url"))

SEMANTIC_SCORE_WEIGHT = 0.1


# Hàm lấy embedding
def get_embedding(text, model=model_name):
    response = client.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding)


def calculate_semantic_score(query: str, model_des: str):
    emb1 = get_embedding(query)
    emb2 = get_embedding(model_des)

    # Tính cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    return similarity


def calculate_dataset_score(query: str, datasets: str):
    emb1 = get_embedding(query)
    emb2 = get_embedding(datasets)

    # Tính cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    return similarity


def calculate_final_score(query: str, model_id: str):
    model_des = model_storage.get_model_by_id(model_id)
    datasets_list = ""
    marker = "- Trained on Datasets: "

    if (
        model_des["documents"]
        and model_des["documents"][0]
        and marker in model_des["documents"][0]
    ):
        pattern = re.escape(marker) + r".*?(\[.*?\])"
        match = re.search(pattern, model_des["documents"][0], flags=re.S)
        if match:
            datasets_list = ast.literal_eval(match.group(0)[23:])
    return SEMANTIC_SCORE_WEIGHT * calculate_semantic_score(
        query, model_des["documents"][0]
    ) + (1 - SEMANTIC_SCORE_WEIGHT) * calculate_dataset_score(
        query, " ".join(datasets_list)
    )


if __name__ == "__main__":
    query = """
    Given an image, classify which category it belongs to from the 1000 ImageNet categories. Output the category name (e.g., "goldfish", "airliner", "koala", etc.)
    """
    model_id_1 = "nateraw/vit-base-patch16-224-cifar10"
    model_id_2 = "farleyknight/mnist-digit-classification-2022-09-04"
    print(calculate_final_score(query, model_id_1))
    print(calculate_final_score(query, model_id_2))
