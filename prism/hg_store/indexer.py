import os
import json
from typing import List, Tuple, Set

from utils.logs import logger
from utils.constants import ROOT_PATH

from prism.hg_store.store import EmbeddingStore
from prism.prompts.prompt_template_manager import PromptTemplateManager
from prism.hg_store.information_extractor import InformationExtractor
from prism.hg_store.typing import HuggingFaceItem
from prism.settings import hg_store_name, prism_embedding_model
class Indexer:

    def __init__(self):
        self.prompt_template_manager = PromptTemplateManager()
        self.embedding_model = prism_embedding_model
        self.store = EmbeddingStore(
            self.embedding_model,
            os.path.join(ROOT_PATH, "cache", "hg_store"),
            32,
            hg_store_name
        )
        self.information_extracter = InformationExtractor()
        self.save_file = "data/processed_huggingface_models.jsonl"
    
    
    async def index_hg_items(self, items: List[HuggingFaceItem]):
        logger.info(f"Indexing Documents")

        logger.info(f"Performing Information Extraction")
        item_ids = [item.id for item in items]
        
        existing_items = self.load_existing_extracted_information()
        
        needed_existing_item = [item for item in existing_items if item.id in item_ids]
        needed_existing_item_ids = [item.id for item in needed_existing_item]
        
        process_items = [item for item in items if item.id not in needed_existing_item_ids]
        if process_items != []:
            process_items = await self.information_extracter.batch_information_extraction(process_items)

        await self.store.insert_hg_items(needed_existing_item + process_items)
        
        self.save_processed_hg_items(process_items)


    def save_processed_hg_items(self, items: List[HuggingFaceItem]):
        asdict_items = [item.to_dict() for item in items]
        with open(self.save_file, "a", encoding="utf-8") as f:
            for item in asdict_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
    def load_existing_extracted_information(self) -> List[HuggingFaceItem]:
        if not os.path.exists(self.save_file):
            return []
        hg_items = []
        with open(self.save_file, "r", encoding="utf-8") as f:
            for line in f:
                hg_items.append(HuggingFaceItem(**json.loads(line)))
        return hg_items
