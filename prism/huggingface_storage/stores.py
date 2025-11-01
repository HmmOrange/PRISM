import os
import json
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
import numpy as np
from collections import defaultdict

from .typing import Triple, NerRawOutput, TripleRawOutput

logger = logging.getLogger(__name__)

class EmbeddingStore:
    """Quản lý storage cho embeddings của chunks, entities và facts"""
    
    def __init__(self, embedding_model, storage_path: str, batch_size: int, store_type: str):
        self.embedding_model = embedding_model
        self.storage_path = storage_path
        self.batch_size = batch_size
        self.store_type = store_type
        self.text_to_hash_id = {}
        self._id_to_rows = {}
        
        # Tạo thư mục storage nếu chưa tồn tại
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing data
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load dữ liệu có sẵn từ storage"""
        metadata_path = os.path.join(self.storage_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.text_to_hash_id = metadata.get('text_to_hash_id', {})
                self._id_to_rows = metadata.get('id_to_rows', {})
    
    def _save_metadata(self):
        """Lưu metadata vào file"""
        metadata_path = os.path.join(self.storage_path, "metadata.json")
        metadata = {
            'text_to_hash_id': self.text_to_hash_id,
            'id_to_rows': self._id_to_rows
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def insert_strings(self, texts: List[str]):
        """Thêm texts mới vào store"""
        new_texts = []
        for text in texts:
            hash_id = self._compute_hash_id(text)
            if hash_id not in self._id_to_rows:
                self.text_to_hash_id[text] = hash_id
                self._id_to_rows[hash_id] = {
                    'hash_id': hash_id,
                    'content': text
                }
                new_texts.append(text)
        
        if new_texts and self.embedding_model:
            # Encode embeddings cho texts mới
            embeddings = self.embedding_model.batch_encode(new_texts, norm=True)
            embedding_path = os.path.join(self.storage_path, "embeddings.npy")
            
            # Load existing embeddings nếu có
            existing_embeddings = []
            if os.path.exists(embedding_path):
                existing_embeddings = np.load(embedding_path)
            
            # Combine embeddings
            if len(existing_embeddings) > 0:
                all_embeddings = np.vstack([existing_embeddings, embeddings])
            else:
                all_embeddings = np.array(embeddings)
            
            # Save embeddings
            np.save(embedding_path, all_embeddings)
        
        self._save_metadata()
    
    def get_missing_string_hash_ids(self, texts: List[str]) -> Dict[str, dict]:
        """Lấy hash IDs cho những strings chưa có trong store"""
        missing = {}
        for text in texts:
            hash_id = self._compute_hash_id(text)
            if hash_id not in self._id_to_rows:
                missing[hash_id] = {
                    'hash_id': hash_id,
                    'content': text
                }
        return missing
    
    def get_all_id_to_rows(self) -> Dict[str, dict]:
        """Lấy tất cả mapping từ ID tới rows"""
        return self._id_to_rows.copy()
    
    def get_all_ids(self) -> List[str]:
        """Lấy tất cả IDs"""
        return list(self._id_to_rows.keys())
    
    def get_all_texts(self) -> List[str]:
        """Lấy tất cả texts"""
        return [row['content'] for row in self._id_to_rows.values()]
    
    def get_row(self, row_id: str) -> dict:
        """Lấy row theo ID"""
        return self._id_to_rows.get(row_id, {})
    
    def get_rows(self, row_ids: List[str]) -> Dict[str, dict]:
        """Lấy multiple rows theo IDs"""
        return {row_id: self._id_to_rows.get(row_id, {}) for row_id in row_ids}
    
    def get_embeddings(self, hash_ids: List[str]) -> np.ndarray:
        """Lấy embeddings cho list hash IDs"""
        embedding_path = os.path.join(self.storage_path, "embeddings.npy")
        if not os.path.exists(embedding_path):
            return np.array([])
        
        all_embeddings = np.load(embedding_path)
        all_ids = self.get_all_ids()
        
        indices = []
        for hash_id in hash_ids:
            if hash_id in all_ids:
                indices.append(all_ids.index(hash_id))
        
        if indices:
            return all_embeddings[indices]
        else:
            return np.array([])
    
    def delete(self, hash_ids: List[str]):
        """Xóa entries theo hash IDs"""
        # Remove from mappings
        texts_to_remove = []
        for text, hash_id in self.text_to_hash_id.items():
            if hash_id in hash_ids:
                texts_to_remove.append(text)
        
        for text in texts_to_remove:
            del self.text_to_hash_id[text]
        
        for hash_id in hash_ids:
            if hash_id in self._id_to_rows:
                del self._id_to_rows[hash_id]
        
        # TODO: Update embeddings array - cần implement logic xóa embeddings tương ứng
        self._save_metadata()
    
    def _compute_hash_id(self, content: str) -> str:
        """Compute hash ID cho content"""
        import hashlib
        prefix = f"{self.store_type}-"
        hash_obj = hashlib.md5(content.encode('utf-8'))
        return prefix + hash_obj.hexdigest()


class OpenIEResultsManager:
    """Quản lý kết quả OpenIE extraction"""
    
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.results = []
        self._load_existing_results()
    
    def _load_existing_results(self):
        """Load kết quả OpenIE có sẵn"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.results = data.get('docs', [])
            except Exception as e:
                logger.warning(f"Cannot load existing OpenIE results: {e}")
                self.results = []
    
    def get_existing_chunk_keys(self) -> Set[str]:
        """Lấy set các chunk keys đã có"""
        return {doc['idx'] for doc in self.results}
    
    def add_results(self, chunk_key: str, passage: str, 
                   entities: List[str], triples: List[Triple]):
        """Thêm kết quả OpenIE mới"""
        result = {
            'idx': chunk_key,
            'passage': passage,
            'extracted_entities': entities,
            'extracted_triples': triples
        }
        self.results.append(result)
    
    def merge_results(self, chunks_to_save: Dict[str, dict],
                     ner_results: Dict[str, NerRawOutput],
                     triple_results: Dict[str, TripleRawOutput]):
        """Merge kết quả OpenIE mới"""
        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            entities = ner_results[chunk_key].unique_entities
            triples = triple_results[chunk_key].triples
            self.add_results(chunk_key, passage, entities, triples)
    
    def save_results(self):
        """Lưu kết quả vào file"""
        if not self.results:
            return
        
        # Tính toán statistics
        sum_phrase_chars = sum([len(e) for doc in self.results for e in doc['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for doc in self.results for e in doc['extracted_entities']])
        num_phrases = sum([len(doc['extracted_entities']) for doc in self.results])
        
        avg_ent_chars = round(sum_phrase_chars / num_phrases, 4) if num_phrases > 0 else 0
        avg_ent_words = round(sum_phrase_words / num_phrases, 4) if num_phrases > 0 else 0
        
        data = {
            'docs': self.results,
            'avg_ent_chars': avg_ent_chars,
            'avg_ent_words': avg_ent_words
        }
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"OpenIE results saved to {self.save_path}")
    
    def get_all_results(self) -> List[dict]:
        """Lấy tất cả kết quả"""
        return self.results.copy()
    
    def filter_results(self, chunk_ids_to_keep: Set[str]):
        """Lọc giữ lại chỉ những results có chunk_id trong set"""
        self.results = [doc for doc in self.results if doc['idx'] in chunk_ids_to_keep]


def reformat_openie_results(all_openie_info: List[dict]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
    """Convert OpenIE results thành format chuẩn"""
    ner_results = {}
    triple_results = {}
    
    for doc in all_openie_info:
        chunk_id = doc['idx']
        
        ner_results[chunk_id] = NerRawOutput(
            chunk_id=chunk_id,
            response=None,
            metadata={},
            unique_entities=doc.get('extracted_entities', [])
        )
        
        triple_results[chunk_id] = TripleRawOutput(
            chunk_id=chunk_id,
            response=None,
            metadata={},
            triples=doc.get('extracted_triples', [])
        )
    
    return ner_results, triple_results
