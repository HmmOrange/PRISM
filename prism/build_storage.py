import os
import json
import asyncio
from utils.constants import ROOT_PATH

from prism.hg_store.indexer import Indexer
from prism.hg_store.typing import HuggingFaceItem
from prism.settings import model_pool_size
async def main():
    indexer = Indexer()
    hg_items = []
    if model_pool_size == "full":
        hg_models_file = os.path.join(ROOT_PATH, "data", "huggingface_models.jsonl")
    elif model_pool_size == "500" or model_pool_size == 500:
        hg_models_file = os.path.join(ROOT_PATH, "data", "huggingface_models_500.jsonl")
    elif model_pool_size == "300" or model_pool_size == 300:
        hg_models_file = os.path.join(ROOT_PATH, "data", "huggingface_models_300.jsonl")
    elif model_pool_size == "perfect":
        hg_models_file = os.path.join(ROOT_PATH, "data", "huggingface_models_perfect.jsonl")
    else:
        raise ValueError(f"model_pool_size {model_pool_size} is not supported")
    with open(hg_models_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            info = json.loads(line)
            hg_items.append(HuggingFaceItem(**info))
    await indexer.index_hg_items(hg_items)
        
if __name__ == '__main__':
    asyncio.run(main())