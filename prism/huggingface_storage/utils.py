import hashlib
import re
import logging
from typing import List, Tuple, Dict, Any, Set
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Tính toán MD5 hash ID cho content với prefix tùy chọn
    
    Args:
        content: Nội dung cần hash
        prefix: Prefix để thêm vào trước hash
    
    Returns:
        Hash ID string
    """
    hash_obj = hashlib.md5(content.encode('utf-8'))
    return prefix + hash_obj.hexdigest()

def text_processing(text: str) -> str:
    """
    Xử lý text cơ bản: chuyển thành lowercase và loại bỏ khoảng trắng thừa
    
    Args:
        text: Text cần xử lý
    
    Returns:
        Text đã được xử lý
    """
    if isinstance(text, (list, tuple)):
        return [text_processing(t) for t in text]
    
    if not isinstance(text, str):
        text = str(text)
    
    # Chuyển về lowercase và loại bỏ khoảng trắng thừa
    processed = text.lower().strip()
    
    # Loại bỏ các ký tự đặc biệt không cần thiết
    processed = re.sub(r'\s+', ' ', processed)
    
    return processed

def extract_entity_nodes(chunk_triples: List[List[Tuple[str, str, str]]]) -> Tuple[List[str], List[List[str]]]:
    """
    Trích xuất entity nodes từ chunk triples
    
    Args:
        chunk_triples: List các chunk, mỗi chunk chứa list các triples
    
    Returns:
        Tuple gồm (unique_entities, chunk_entities_list)
    """
    all_entities = set()
    chunk_entities_list = []
    
    for chunk_triples_list in chunk_triples:
        chunk_entities = set()
        
        for triple in chunk_triples_list:
            if len(triple) >= 3:
                # Subject và Object entities
                subject = str(triple[0]).strip()
                obj = str(triple[2]).strip()
                
                if subject:
                    all_entities.add(subject)
                    chunk_entities.add(subject)
                if obj:
                    all_entities.add(obj)
                    chunk_entities.add(obj)
        
        chunk_entities_list.append(list(chunk_entities))
    
    return list(all_entities), chunk_entities_list

def flatten_facts(chunk_triples: List[List[Tuple[str, str, str]]]) -> List[Tuple[str, str, str]]:
    """
    Flatten danh sách chunk triples thành một list các triples
    
    Args:
        chunk_triples: List các chunk, mỗi chunk chứa list các triples
    
    Returns:
        List tất cả triples đã được flatten
    """
    flattened = []
    
    for chunk_triples_list in chunk_triples:
        for triple in chunk_triples_list:
            if len(triple) >= 3:
                flattened.append(tuple(triple[:3]))
    
    return flattened

def min_max_normalize(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize scores về khoảng [0, 1] sử dụng min-max normalization
    
    Args:
        scores: Array của scores cần normalize
        eps: Epsilon để tránh chia cho 0
    
    Returns:
        Normalized scores
    """
    if len(scores) == 0:
        return scores
    
    scores = np.array(scores)
    
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score - min_score < eps:
        # Nếu tất cả scores giống nhau, return uniform distribution
        return np.ones_like(scores) / len(scores) if len(scores) > 0 else scores
    
    normalized = (scores - min_score) / (max_score - min_score)
    return normalized

def retrieve_knn(query_ids: List[str], 
                key_ids: List[str], 
                query_vecs: np.ndarray, 
                key_vecs: np.ndarray, 
                k: int = 10,
                query_batch_size: int = 64,
                key_batch_size: int = 64) -> Dict[str, Tuple[List[str], List[float]]]:
    """
    Thực hiện KNN search để tìm k nearest neighbors
    
    Args:
        query_ids: List query IDs
        key_ids: List key IDs
        query_vecs: Query vectors
        key_vecs: Key vectors
        k: Số lượng nearest neighbors
        query_batch_size: Batch size cho queries
        key_batch_size: Batch size cho keys
    
    Returns:
        Dict mapping query_id -> (neighbor_ids, similarity_scores)
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    if len(query_vecs) == 0 or len(key_vecs) == 0:
        return {}
    
    results = {}
    
    # Ensure vectors are 2D
    if query_vecs.ndim == 1:
        query_vecs = query_vecs.reshape(1, -1)
    if key_vecs.ndim == 1:
        key_vecs = key_vecs.reshape(1, -1)
    
    # Compute similarity matrix
    try:
        similarity_matrix = cosine_similarity(query_vecs, key_vecs)
        
        for i, query_id in enumerate(query_ids):
            if i < len(similarity_matrix):
                scores = similarity_matrix[i]
                
                # Get top k indices (excluding self if query_id in key_ids)
                top_k_indices = np.argsort(scores)[::-1]
                
                neighbor_ids = []
                neighbor_scores = []
                
                for idx in top_k_indices:
                    if len(neighbor_ids) >= k:
                        break
                    
                    if idx < len(key_ids):
                        neighbor_id = key_ids[idx]
                        score = scores[idx]
                        
                        # Skip self-similarity if needed
                        if neighbor_id != query_id:
                            neighbor_ids.append(neighbor_id)
                            neighbor_scores.append(float(score))
                
                # Pad to k if necessary
                while len(neighbor_ids) < k and len(neighbor_ids) < len(key_ids):
                    neighbor_ids.append("")
                    neighbor_scores.append(0.0)
                
                results[query_id] = (neighbor_ids[:k], neighbor_scores[:k])
        
    except Exception as e:
        logger.error(f"Error in KNN retrieval: {e}")
        # Return empty results for all queries
        for query_id in query_ids:
            results[query_id] = ([], [])
    
    return results

def batch_process(items: List[Any], batch_size: int = 32):
    """
    Generator để xử lý items theo batch
    
    Args:
        items: List items cần xử lý
        batch_size: Kích thước batch
    
    Yields:
        Batch của items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Chia an toàn, trả về default nếu b = 0
    
    Args:
        a: Số bị chia
        b: Số chia
        default: Giá trị trả về nếu b = 0
    
    Returns:
        Kết quả chia hoặc default
    """
    return a / b if b != 0 else default

def clean_text(text: str) -> str:
    """
    Làm sạch text bằng cách loại bỏ ký tự đặc biệt và chuẩn hóa
    
    Args:
        text: Text cần làm sạch
    
    Returns:
        Text đã được làm sạch
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Loại bỏ các ký tự không phải alphanumeric và space
    cleaned = re.sub(r'[^\w\s]', ' ', text)
    
    # Loại bỏ khoảng trắng thừa
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def compute_entity_hash(entity: str) -> str:
    """
    Tính hash cho entity với prefix "entity-"
    
    Args:
        entity: Entity string
    
    Returns:
        Hash ID với prefix
    """
    return compute_mdhash_id(content=entity, prefix="entity-")

def compute_chunk_hash(chunk: str) -> str:
    """
    Tính hash cho chunk với prefix "chunk-"
    
    Args:
        chunk: Chunk string
    
    Returns:
        Hash ID với prefix
    """
    return compute_mdhash_id(content=chunk, prefix="chunk-")
