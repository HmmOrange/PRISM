"""
HippoRAG Implementation - Modular version

Chia nhỏ code HippoRAG thành các modules có tính logic:
- typing: Định nghĩa data types và configs
- stores: Quản lý embedding storage và OpenIE results
- utils: Utility functions
- graph: Knowledge graph operations
- indexing: Document indexing operations  
- retrieval: Retrieval và QA operations

Usage:
    from huggingface_storage import HippoRAG, RetrievalConfig
    
    config = RetrievalConfig(retrieval_top_k=20)
    hippo = HippoRAG(working_dir="./data", config=config)
    
    # Index documents
    hippo.index_documents(["Document 1", "Document 2"])
    
    # Perform retrieval
    results = hippo.retrieve(["Query 1", "Query 2"])
"""

from .typing import (
    QuerySolution, NerRawOutput, TripleRawOutput, 
    GraphInfo, RetrievalConfig, IndexingResult, Triple
)

from .stores import (
    EmbeddingStore, OpenIEResultsManager, reformat_openie_results
)

from .utils import (
    compute_mdhash_id, text_processing, extract_entity_nodes,
    flatten_facts, min_max_normalize, retrieve_knn
)

from .graph import KnowledgeGraph

from .indexing import HippoIndexer

from .retrieval import HippoRetriever

# Main class được implement trong file riêng
# from .hippo_rag import HippoRAG

__all__ = [
    # Data types
    'QuerySolution', 'NerRawOutput', 'TripleRawOutput', 
    'GraphInfo', 'RetrievalConfig', 'IndexingResult', 'Triple',
    
    # Storage classes
    'EmbeddingStore', 'OpenIEResultsManager', 'reformat_openie_results',
    
    # Utility functions
    'compute_mdhash_id', 'text_processing', 'extract_entity_nodes',
    'flatten_facts', 'min_max_normalize', 'retrieve_knn',
    
    # Core classes
    'KnowledgeGraph', 'HippoIndexer', 'HippoRetriever',
    
    # Main class
    # 'HippoRAG'
] 