from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Set, Union
import numpy as np

# Type aliases
Triple = Tuple[str, str, str]

@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_scores: List[float]
    answer: str = ""
    gold_answers: Optional[List[str]] = None
    gold_docs: Optional[List[str]] = None

@dataclass
class NerRawOutput:
    chunk_id: str
    response: Optional[str]
    metadata: Dict[str, Any]
    unique_entities: List[str]

@dataclass
class TripleRawOutput:
    chunk_id: str
    response: Optional[str]
    metadata: Dict[str, Any]
    triples: List[Triple]

@dataclass
class GraphInfo:
    """Thông tin thống kê về graph"""
    num_phrase_nodes: int
    num_passage_nodes: int
    num_total_nodes: int
    num_extracted_triples: int
    num_triples_with_passage_node: int
    num_synonymy_triples: int
    num_total_triples: int

@dataclass
class RetrievalConfig:
    """Cấu hình cho retrieval operations"""
    retrieval_top_k: int = 10
    linking_top_k: int = 5
    qa_top_k: int = 5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    synonymy_edge_topk: int = 10
    synonymy_edge_sim_threshold: float = 0.8
    synonymy_edge_query_batch_size: int = 64
    synonymy_edge_key_batch_size: int = 64
    embedding_batch_size: int = 32

@dataclass
class IndexingResult:
    """Kết quả của quá trình indexing"""
    num_docs_indexed: int
    num_entities_extracted: int
    num_triples_extracted: int
    processing_time: float
    graph_info: GraphInfo