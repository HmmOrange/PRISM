import os
import pandas as pd
import numpy as np
from typing import List
from copy import deepcopy

from utils.logs import logger
from provider.embedding.base_embedding import BaseEmbedding


from prism.hg_store.utils import compute_mdhash_id
from prism.hg_store.typing import HuggingFaceItem

class EmbeddingStore:
    
    def __init__(self, embedding_model: BaseEmbedding, db_filename: str, batch_size: int, namespace: str):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace
        
        if not os.path.exists(db_filename):
            logger.info(f"Creating directory {db_filename}")
            os.makedirs(db_filename, exist_ok=True)
        
        self.filename = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        self._load_data()
        
    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids, self.texts, self.embeddings, self.task_types, self.metadatas, self.classification_embeddings, self.tabular_embeddings = df["hash_id"].values.tolist(), df["content"].values.tolist(), df["embedding"].values.tolist(), df["task_type"].values.tolist(), df["metadata"].values.tolist(), df["classification_embedding"].values.tolist(), df["tabular_embedding"].values.tolist()
            
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t, "task_type": task_type, "embedding": embedding, "metadata": metadata, "classification_embedding": classification_embedding, "tabular_embedding": tabular_embedding}
                for h, t, task_type, embedding, metadata, classification_embedding, tabular_embedding in zip(self.hash_ids, self.texts, self.task_types, self.embeddings, self.metadatas, self.classification_embeddings, self.tabular_embeddings)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h  for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings, self.task_types, self.metadatas, self.classification_embeddings, self.tabular_embeddings = [], [], [], [], [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}
            
    def _save_data(self):
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings,
            "task_type": self.task_types,
            "metadata": self.metadatas,
            "classification_embedding": self.classification_embeddings,
            "tabular_embedding": self.tabular_embeddings
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {h: {"hash_id": h, "content": t, "task_types": task_types, "metadatas": metadatas, "classification_embeddings": classification_embeddings, "tabular_embeddings": tabular_embeddings} for h, t, e, task_types, metadatas, classification_embeddings, tabular_embeddings in zip(self.hash_ids, self.texts, self.embeddings, self.task_types, self.metadatas, self.classification_embeddings, self.tabular_embeddings)}
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, metadatas, task_types, embeddings, classification_embeddings, tabular_embeddings):
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.task_types.extend(task_types)
        self.embeddings.extend(embeddings)
        self.classification_embeddings.extend(classification_embeddings)
        self.tabular_embeddings.extend(tabular_embeddings)

        logger.info(f"Saving new records.")
        self._save_data()

    def delete(self, hash_ids: list[str]):
        indices = []

        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)
            self.task_types.pop(idx)
            self.metadatas.pop(idx)
            self.classification_embeddings.pop(idx)
            self.tabular_embeddings.pop(idx)

        logger.info(f"Saving record after deletion.")
        self._save_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text):
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id : self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self):
        return set(row['content'] for row in self.hash_id_to_row.values())

    def get_all_rows(self):
        return deepcopy(self.hash_id_to_row)
    
    def get_rows_by_task_type(self, task_types: List[str]):
        return [row for row in self.hash_id_to_row.values() if row['task_type'] in task_types]

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)
    
    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return [embeddings[i] for i in range(len(embeddings))]
    
    def get_missing_string_hash_ids(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  {}

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    async def insert_hg_items(self, hg_items: List[HuggingFaceItem]):
        # Get all hash_ids from the input dictionary.
        all_hash_ids = [item.id for item in hg_items]
        if not all_hash_ids:
            return  # Nothing to insert.

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        missing_items = [item for item in hg_items if item.id in missing_ids]
        
        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist.")

        if not missing_ids:
            return {} # All records already exist.

        # Prepare the texts to encode from the "content" field.
        texts_to_encode = [item.summary for item in missing_items if item.summary]
        assert len(texts_to_encode) == len(missing_items)
        
        metadatas = [item.entity_extracted for item in missing_items if item.entity_extracted]
        assert len(metadatas) == len(missing_items)
        
        task_types = [item.pipeline_tag for item in missing_items]
        
        classification_contents_to_encode = [f"Datasets: {', '.join(item.entity_extracted.get('datasets', ''))}. Classes: {', '.join(item.entity_extracted.get('classes', ''))}. Number of classes: {item.entity_extracted.get('num_classes', '')}" for item in missing_items if item.entity_extracted]
        assert len(classification_contents_to_encode) == len(missing_items)
        
        tabular_contents_to_encode = [f"Input format: {item.entity_extracted.get('input_format', '')}" for item in missing_items if item.entity_extracted]
        assert len(tabular_contents_to_encode) == len(missing_items)
        
        missing_embeddings = await self.embedding_model.batch_encode(texts_to_encode)
        missing_classification_embeddings = await self.embedding_model.batch_encode(classification_contents_to_encode)
        missing_tabular_embeddings = await self.embedding_model.batch_encode(tabular_contents_to_encode)
        
        self._upsert(hash_ids=missing_ids, texts=texts_to_encode, metadatas=metadatas, task_types=task_types, embeddings=missing_embeddings, classification_embeddings=missing_classification_embeddings, tabular_embeddings=missing_tabular_embeddings)
        
        
