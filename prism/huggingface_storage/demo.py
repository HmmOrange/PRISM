"""
Demo script để minh họa cách sử dụng HippoRAG modular implementation

Chạy script này để xem cách các modules hoạt động cùng nhau
"""

import os
import logging
from typing import List

# Import các modules đã tạo
from .typing import RetrievalConfig, QuerySolution
from .indexing import HippoIndexer
from .retrieval import HippoRetriever

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockEmbeddingModel:
    """Mock embedding model cho demo"""
    
    def batch_encode(self, texts: List[str], instruction: str = None, norm: bool = True):
        """Tạo fake embeddings cho demo"""
        import numpy as np
        # Tạo random embeddings với dimension 768
        embeddings = np.random.randn(len(texts), 768)
        if norm:
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        return embeddings

class MockOpenIEModel:
    """Mock OpenIE model cho demo"""
    
    def batch_openie(self, chunks_to_process: dict):
        """Tạo fake OpenIE results cho demo"""
        from .typing import NerRawOutput, TripleRawOutput
        
        ner_results = {}
        triple_results = {}
        
        for chunk_id, row in chunks_to_process.items():
            content = row['content']
            
            # Tạo fake entities (lấy một số từ từ content)
            words = content.split()[:5]  # Lấy 5 từ đầu làm entities
            entities = [word.strip('.,!?') for word in words if len(word) > 2]
            
            ner_results[chunk_id] = NerRawOutput(
                chunk_id=chunk_id,
                response=f"Extracted entities from: {content[:50]}...",
                metadata={"model": "mock", "confidence": 0.8},
                unique_entities=entities
            )
            
            # Tạo fake triples
            triples = []
            if len(entities) >= 3:
                # Tạo một vài triples đơn giản
                triples.append((entities[0], "relates_to", entities[1]))
                if len(entities) >= 3:
                    triples.append((entities[1], "connects_with", entities[2]))
            
            triple_results[chunk_id] = TripleRawOutput(
                chunk_id=chunk_id,
                response=f"Extracted triples from: {content[:50]}...",
                metadata={"model": "mock", "confidence": 0.7},
                triples=triples
            )
        
        return ner_results, triple_results

class MockLLMModel:
    """Mock LLM model cho QA demo"""
    
    def infer(self, messages):
        """Tạo fake answer cho demo"""
        # Extract question từ messages
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Tạo simple answer
        if "Question:" in user_message:
            question = user_message.split("Question:")[-1].strip()
            answer = f"Based on the provided documents, the answer to '{question}' is a demonstration response from the mock LLM model."
        else:
            answer = "This is a demonstration response from the mock LLM model."
        
        return answer, {"tokens": 50, "time": 0.1}, False

def demo_hippo_rag():
    """Demo chính cho HippoRAG modular implementation"""
    
    logger.info("=== HippoRAG Modular Implementation Demo ===")
    
    # 1. Setup configuration
    config = RetrievalConfig(
        retrieval_top_k=5,
        linking_top_k=3,
        qa_top_k=3,
        embedding_batch_size=16
    )
    
    logger.info(f"Configuration: {config}")
    
    # 2. Setup models (mock models cho demo)
    embedding_model = MockEmbeddingModel()
    openie_model = MockOpenIEModel()
    llm_model = MockLLMModel()
    
    working_dir = "./demo_hippo_storage"
    os.makedirs(working_dir, exist_ok=True)
    
    # 3. Initialize Indexer
    logger.info("\n--- Initializing Indexer ---")
    indexer = HippoIndexer(
        working_dir=working_dir,
        openie_model=openie_model,
        embedding_model=embedding_model,
        config=config,
        force_rebuild=False
    )
    
    # 4. Sample documents để index
    sample_documents = [
        "Artificial intelligence is transforming the way we work and live. Machine learning algorithms can analyze vast amounts of data.",
        "Natural language processing enables computers to understand human language. Deep learning models have achieved remarkable results.",
        "Computer vision allows machines to interpret visual information. Neural networks are inspired by the human brain structure.",
        "Robotics combines AI with mechanical engineering. Autonomous vehicles use sensors and AI to navigate safely.",
        "Data science involves extracting insights from data. Statistics and programming skills are essential for data scientists."
    ]
    
    # 5. Index documents
    logger.info("\n--- Indexing Documents ---")
    indexing_result = indexer.index_documents(sample_documents)
    
    logger.info(f"Indexing Result:")
    logger.info(f"  - Documents indexed: {indexing_result.num_docs_indexed}")
    logger.info(f"  - Entities extracted: {indexing_result.num_entities_extracted}")
    logger.info(f"  - Triples extracted: {indexing_result.num_triples_extracted}")
    logger.info(f"  - Processing time: {indexing_result.processing_time:.2f}s")
    logger.info(f"  - Graph info: {indexing_result.graph_info}")
    
    # 6. Check if ready for retrieval
    if not indexer.is_ready_for_retrieval():
        logger.warning("Indexer not ready for retrieval!")
        return
    
    # 7. Initialize Retriever
    logger.info("\n--- Initializing Retriever ---")
    embedding_stores = indexer.get_embedding_stores()
    knowledge_graph = indexer.get_knowledge_graph()
    
    retriever = HippoRetriever(
        knowledge_graph=knowledge_graph,
        embedding_stores=embedding_stores,
        llm_model=llm_model,
        rerank_filter=None,  # No rerank filter trong demo
        config=config
    )
    
    # 8. Sample queries
    sample_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?"
    ]
    
    # 9. Perform retrieval
    logger.info("\n--- Performing Retrieval ---")
    query_solutions = retriever.retrieve(sample_queries, embedding_model)
    
    for i, solution in enumerate(query_solutions):
        logger.info(f"\nQuery {i+1}: {solution.question}")
        logger.info(f"Retrieved {len(solution.docs)} documents:")
        for j, doc in enumerate(solution.docs[:2]):  # Show only first 2 docs
            logger.info(f"  Doc {j+1}: {doc[:100]}..." if len(doc) > 100 else f"  Doc {j+1}: {doc}")
        logger.info(f"Scores: {solution.doc_scores[:2]}")
    
    # 10. Perform QA
    logger.info("\n--- Performing QA ---")
    qa_solutions, responses, metadata = retriever.qa(query_solutions)
    
    for i, solution in enumerate(qa_solutions):
        logger.info(f"\nQ: {solution.question}")
        logger.info(f"A: {solution.answer}")
    
    # 11. Performance statistics
    logger.info("\n--- Performance Statistics ---")
    logger.info(f"Total retrieval time: {retriever.total_retrieval_time:.2f}s")
    logger.info(f"PPR time: {retriever.ppr_time:.2f}s")
    logger.info(f"Rerank time: {retriever.rerank_time:.2f}s")
    
    # 12. Demo document deletion
    logger.info("\n--- Testing Document Deletion ---")
    documents_to_delete = [sample_documents[0]]  # Delete first document
    success = indexer.delete_documents(documents_to_delete)
    logger.info(f"Document deletion success: {success}")
    
    logger.info("\n=== Demo Completed Successfully! ===")
    
    # Cleanup
    import shutil
    try:
        shutil.rmtree(working_dir)
        logger.info("Cleaned up demo files")
    except:
        logger.warning("Could not clean up demo files")

if __name__ == "__main__":
    demo_hippo_rag() 