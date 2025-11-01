import logging
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm

from .typing import QuerySolution, RetrievalConfig
from .utils import min_max_normalize, compute_mdhash_id

logger = logging.getLogger(__name__)

class HippoRetriever:
    """Retrieval engine cho HippoRAG"""
    
    def __init__(self, knowledge_graph, embedding_stores: Dict[str, Any], 
                 llm_model=None, rerank_filter=None, config: RetrievalConfig = None):
        """
        Args:
            knowledge_graph: KnowledgeGraph instance
            embedding_stores: Dict chứa chunk_store, entity_store, fact_store
            llm_model: Language model cho QA
            rerank_filter: Filter cho reranking facts
            config: Retrieval configuration
        """
        self.knowledge_graph = knowledge_graph
        self.chunk_store = embedding_stores.get('chunk_store')
        self.entity_store = embedding_stores.get('entity_store')
        self.fact_store = embedding_stores.get('fact_store')
        self.llm_model = llm_model
        self.rerank_filter = rerank_filter
        self.config = config if config is not None else RetrievalConfig()
        
        # Embeddings cache
        self.query_embeddings_cache: Dict[str, Dict[str, np.ndarray]] = {
            'triple': {},
            'passage': {}
        }
        
        # Performance tracking
        self.ppr_time = 0.0
        self.rerank_time = 0.0
        self.total_retrieval_time = 0.0
        
        # Precomputed embeddings
        self.entity_embeddings: Optional[np.ndarray] = None
        self.passage_embeddings: Optional[np.ndarray] = None
        self.fact_embeddings: Optional[np.ndarray] = None
        
        # Facts to docs mapping
        self.proc_triples_to_docs: Dict[str, set] = {}
        
        self._prepare_retrieval_data()
    
    def _prepare_retrieval_data(self):
        """Chuẩn bị dữ liệu cho retrieval"""
        logger.info("Preparing retrieval data...")
        
        # Load embeddings
        if self.entity_store:
            entity_keys = self.knowledge_graph.entity_node_keys
            self.entity_embeddings = self.entity_store.get_embeddings(entity_keys)
        
        if self.chunk_store:
            passage_keys = self.knowledge_graph.passage_node_keys
            self.passage_embeddings = self.chunk_store.get_embeddings(passage_keys)
        
        if self.fact_store:
            fact_keys = self.fact_store.get_all_ids()
            self.fact_embeddings = self.fact_store.get_embeddings(fact_keys)
        
        logger.info("Retrieval data preparation completed")
    
    def get_query_embeddings(self, queries: List[str], embedding_model):
        """Lấy embeddings cho queries"""
        new_queries = []
        
        for query in queries:
            if query not in self.query_embeddings_cache['triple'] or \
               query not in self.query_embeddings_cache['passage']:
                new_queries.append(query)
        
        if new_queries:
            # Get embeddings for fact retrieval
            logger.info(f"Encoding {len(new_queries)} queries for fact retrieval")
            fact_embeddings = embedding_model.batch_encode(
                new_queries, 
                instruction="query_to_fact",  # You may need to implement this
                norm=True
            )
            
            for query, embedding in zip(new_queries, fact_embeddings):
                self.query_embeddings_cache['triple'][query] = embedding
            
            # Get embeddings for passage retrieval
            logger.info(f"Encoding {len(new_queries)} queries for passage retrieval")
            passage_embeddings = embedding_model.batch_encode(
                new_queries,
                instruction="query_to_passage",  # You may need to implement this
                norm=True
            )
            
            for query, embedding in zip(new_queries, passage_embeddings):
                self.query_embeddings_cache['passage'][query] = embedding
    
    def get_fact_scores(self, query: str) -> np.ndarray:
        """Tính similarity scores giữa query và facts"""
        query_embedding = self.query_embeddings_cache['triple'].get(query)
        
        if query_embedding is None or self.fact_embeddings is None or len(self.fact_embeddings) == 0:
            logger.warning("No query embedding or fact embeddings available")
            return np.array([])
        
        try:
            # Compute similarity
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T)
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {e}")
            return np.array([])
    
    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[tuple], dict]:
        """Rerank facts dựa trên query relevance"""
        link_top_k = self.config.linking_top_k
        
        if len(query_fact_scores) == 0:
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
        
        try:
            # Get top facts by score
            if len(query_fact_scores) <= link_top_k:
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
            
                         # Get actual facts
            if self.fact_store is None:
                return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': 'No fact store'}
            
            fact_keys = self.fact_store.get_all_ids()
            candidate_fact_ids = [fact_keys[idx] for idx in candidate_fact_indices]
            fact_rows = self.fact_store.get_rows(candidate_fact_ids)
            candidate_facts = [eval(fact_rows[fid]['content']) for fid in candidate_fact_ids]
            
            # Apply reranking if available
            if self.rerank_filter:
                top_k_fact_indices, top_k_facts, _ = self.rerank_filter(
                    query, candidate_facts, candidate_fact_indices, len_after_rerank=link_top_k
                )
            else:
                top_k_fact_indices = candidate_fact_indices
                top_k_facts = candidate_facts
            
            rerank_log = {
                'facts_before_rerank': candidate_facts,
                'facts_after_rerank': top_k_facts
            }
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {e}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}
    
    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """Dense passage retrieval"""
        query_embedding = self.query_embeddings_cache['passage'].get(query)
        
        if query_embedding is None or self.passage_embeddings is None:
            logger.warning("No query embedding or passage embeddings available")
            return np.array([]), np.array([])
        
        try:
            # Compute similarity scores
            query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
            query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
            query_doc_scores = min_max_normalize(query_doc_scores)
            
            # Sort by scores
            sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
            sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
            
            return sorted_doc_ids, sorted_doc_scores
            
        except Exception as e:
            logger.error(f"Error in dense passage retrieval: {e}")
            return np.array([]), np.array([])
    
    def graph_search_with_facts(self, query: str, top_k_facts: List[tuple], 
                               top_k_fact_indices: List[int], query_fact_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Graph search sử dụng fact entities và PPR"""
        
        # Initialize weights
        phrase_weights = np.zeros(len(self.knowledge_graph.entity_node_keys))
        passage_weights = np.zeros(len(self.knowledge_graph.passage_node_keys))
        
        # Compute phrase weights from facts
        linking_score_map = {}
        phrase_scores = {}
        
        for rank, fact in enumerate(top_k_facts):
            if len(fact) >= 3:
                subject_phrase = fact[0].lower()
                object_phrase = fact[2].lower()
                fact_score = query_fact_scores[top_k_fact_indices[rank]] if len(query_fact_scores) > 0 else 0.0
                
                for phrase in [subject_phrase, object_phrase]:
                    phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                    
                    # Find phrase index in entity_node_keys
                    if phrase_key in self.knowledge_graph.entity_node_keys:
                        phrase_idx = self.knowledge_graph.entity_node_keys.index(phrase_key)
                        
                        # Weight by inverse frequency
                        ent_chunk_count = len(self.knowledge_graph.ent_node_to_chunk_ids.get(phrase_key, set()))
                        weighted_score = fact_score / max(ent_chunk_count, 1)
                        
                        phrase_weights[phrase_idx] += weighted_score
                        
                        if phrase not in phrase_scores:
                            phrase_scores[phrase] = []
                        phrase_scores[phrase].append(weighted_score)
        
        # Compute linking score map
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))
        
        # Get DPR scores for passages
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_scores = min_max_normalize(dpr_sorted_doc_scores)
        
        # Set passage weights
        for i, doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            if i < len(self.knowledge_graph.passage_node_keys) and i < len(normalized_dpr_scores):
                passage_weights[doc_id] = normalized_dpr_scores[i] * self.config.passage_node_weight
        
        # Combine weights for PPR
        node_weights = self.knowledge_graph.get_node_weights_for_ppr(phrase_weights, passage_weights)
        
        if np.sum(node_weights) == 0:
            logger.warning("No valid node weights for PPR, falling back to DPR")
            return dpr_sorted_doc_ids, dpr_sorted_doc_scores
        
        # Run PPR
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.knowledge_graph.run_personalized_pagerank(
            node_weights, damping=self.config.damping
        )
        ppr_end = time.time()
        
        self.ppr_time += (ppr_end - ppr_start)
        
        return ppr_sorted_doc_ids, ppr_sorted_doc_scores
    
    def retrieve(self, queries: List[str], embedding_model, num_to_retrieve: Optional[int] = None) -> List[QuerySolution]:
        """Thực hiện retrieval cho list queries"""
        retrieve_start = time.time()
        
        if num_to_retrieve is None:
            num_to_retrieve = self.config.retrieval_top_k
        
        # Get query embeddings
        self.get_query_embeddings(queries, embedding_model)
        
        results = []
        
        for query in tqdm(queries, desc="Retrieving"):
            # Get fact scores
            query_fact_scores = self.get_fact_scores(query)
            
            # Rerank facts
            rerank_start = time.time()
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            rerank_end = time.time()
            self.rerank_time += (rerank_end - rerank_start)
            
            if len(top_k_facts) == 0:
                logger.info('No facts found after reranking, using DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_facts(
                    query, top_k_facts, top_k_fact_indices, query_fact_scores
                )
            
                         # Get top documents
            top_docs = []
            for doc_id in sorted_doc_ids[:num_to_retrieve]:
                if doc_id < len(self.knowledge_graph.passage_node_keys) and self.chunk_store is not None:
                    passage_key = self.knowledge_graph.passage_node_keys[doc_id]
                    doc_content = self.chunk_store.get_row(passage_key).get("content", "")
                    top_docs.append(doc_content)
            
            results.append(QuerySolution(
                question=query,
                docs=top_docs,
                doc_scores=sorted_doc_scores[:num_to_retrieve].tolist()
            ))
        
        retrieve_end = time.time()
        self.total_retrieval_time += (retrieve_end - retrieve_start)
        
        # Log performance
        logger.info(f"Total Retrieval Time: {self.total_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time: {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time: {self.ppr_time:.2f}s")
        
        return results
    
    def qa(self, query_solutions: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[dict]]:
        """Thực hiện QA trên retrieved documents"""
        if not self.llm_model:
            logger.warning("No LLM model available for QA")
            return query_solutions, [], []
        
        all_qa_messages = []
        
        # Prepare QA prompts
        for query_solution in tqdm(query_solutions, desc="Preparing QA prompts"):
            retrieved_passages = query_solution.docs[:self.config.qa_top_k]
            
            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Document: {passage}\n\n'
            prompt_user += f'Question: {query_solution.question}\nAnswer: '
            
            # Simple prompt format - you may want to customize this
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documents."},
                {"role": "user", "content": prompt_user}
            ]
            all_qa_messages.append(messages)
        
        # Run QA inference
        all_qa_results = []
        for qa_messages in tqdm(all_qa_messages, desc="QA Reading"):
            try:
                result = self.llm_model.infer(qa_messages)
                all_qa_results.append(result)
            except Exception as e:
                logger.error(f"Error in QA inference: {e}")
                all_qa_results.append(("Error occurred", {}, False))
        
        all_response_messages, all_metadata, all_cache_hits = zip(*all_qa_results)
        
        # Extract answers
        updated_solutions = []
        for i, query_solution in enumerate(query_solutions):
            response = all_response_messages[i]
            
            # Simple answer extraction - you may want to improve this
            try:
                if "Answer:" in response:
                    answer = response.split("Answer:")[-1].strip()
                else:
                    answer = response.strip()
            except Exception as e:
                logger.warning(f"Error extracting answer: {e}")
                answer = response
            
            query_solution.answer = answer
            updated_solutions.append(query_solution)
        
        return updated_solutions, list(all_response_messages), list(all_metadata)
    
    def rag_qa(self, queries: List[str], embedding_model, 
               gold_docs: Optional[List[List[str]]] = None,
               gold_answers: Optional[List[List[str]]] = None) -> Tuple[List[QuerySolution], List[str], List[dict]]:
        """Thực hiện RAG QA pipeline hoàn chỉnh"""
        
        # Retrieval
        query_solutions = self.retrieve(queries, embedding_model)
        
        # QA
        query_solutions, response_messages, metadata = self.qa(query_solutions)
        
        # Add gold data if provided
        if gold_answers:
            for i, solution in enumerate(query_solutions):
                if i < len(gold_answers):
                    solution.gold_answers = gold_answers[i]
        
        if gold_docs:
            for i, solution in enumerate(query_solutions):
                if i < len(gold_docs):
                    solution.gold_docs = gold_docs[i]
        
        return query_solutions, response_messages, metadata
    
    def reset_performance_timers(self):
        """Reset performance tracking timers"""
        self.ppr_time = 0.0
        self.rerank_time = 0.0
        self.total_retrieval_time = 0.0
