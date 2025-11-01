from typing import List, Any, Dict, Literal, Optional, Tuple, TypedDict
from dataclasses import dataclass, asdict

import numpy as np

Triple = Tuple[str, str, str]

@dataclass
class HuggingFaceItem:
    id: str
    pipeline_tag: str
    tags: List[str]
    description: str
    downloads: int
    likes: int
    meta: Dict[str, Any]
    inference_type: Literal["huggingface", "local"]
    summary: Optional[str] = None
    entity_extracted: Optional[dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]
    
@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]
    
@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal["node", "dpr"]
    
class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]
    
@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_scores: Optional[np.ndarray] = None
    answer: Optional[str] = None
    gold_answers: Optional[List[str]] = None
    gold_docs: Optional[List[str]] = None
    
    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answer": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]]  if self.doc_scores is not None else None,
            "gold_docs": self.gold_docs,
        }