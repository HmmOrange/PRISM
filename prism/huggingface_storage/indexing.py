import os
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from .typing import IndexingResult, RetrievalConfig
from .stores import EmbeddingStore, OpenIEResultsManager, reformat_openie_results
from .graph import KnowledgeGraph
from .utils import text_processing, extract_entity_nodes, flatten_facts

logger = logging.getLogger(__name__)

class HippoIndexer:
    """Indexing engine cho HippoRAG"""
    
    def __init__(self, working_dir: str, openie_model=None, embedding_model=None, 
                 config: RetrievalConfig = None, force_rebuild: bool = False):
        """
        Args:
            working_dir: Thư mục làm việc để lưu các files
            openie_model: Model cho OpenIE extraction
            embedding_model: Model cho embedding
            config: Configuration object
            force_rebuild: Có rebuild từ đầu không
        """
        self.working_dir = working_dir
        self.openie_model = openie_model
        self.embedding_model = embedding_model
        self.config = config if config is not None else RetrievalConfig()
        self.force_rebuild = force_rebuild
        
        # Create working directory
        os.makedirs(working_dir, exist_ok=True)
        
        # Initialize components
        self._initialize_stores()
        self._initialize_graph()
        self._initialize_openie_manager()
    
    def _initialize_stores(self):
        """Khởi tạo embedding stores"""
        self.chunk_store = EmbeddingStore(
            embedding_model=self.embedding_model,
            storage_path=os.path.join(self.working_dir, "chunk_embeddings"),
            batch_size=self.config.embedding_batch_size,
            store_type="chunk"
        )
        
        self.entity_store = EmbeddingStore(
            embedding_model=self.embedding_model,
            storage_path=os.path.join(self.working_dir, "entity_embeddings"),
            batch_size=self.config.embedding_batch_size,
            store_type="entity"
        )
        
        self.fact_store = EmbeddingStore(
            embedding_model=self.embedding_model,
            storage_path=os.path.join(self.working_dir, "fact_embeddings"),
            batch_size=self.config.embedding_batch_size,
            store_type="fact"
        )
    
    def _initialize_graph(self):
        """Khởi tạo knowledge graph"""
        self.knowledge_graph = KnowledgeGraph(
            working_dir=self.working_dir,
            is_directed=False,  # Default to undirected
            force_rebuild=self.force_rebuild
        )
    
    def _initialize_openie_manager(self):
        """Khởi tạo OpenIE results manager"""
        openie_results_path = os.path.join(self.working_dir, "openie_results.json")
        self.openie_manager = OpenIEResultsManager(openie_results_path)
    
    def index_documents(self, documents: List[str]) -> IndexingResult:
        """
        Index danh sách documents
        
        Args:
            documents: List documents cần index
            
        Returns:
            IndexingResult với thông tin về quá trình indexing
        """
        start_time = time.time()
        
        logger.info(f"Starting indexing of {len(documents)} documents")
        
        # 1. Insert documents vào chunk store
        logger.info("Inserting documents into chunk store...")
        self.chunk_store.insert_strings(documents)
        chunk_to_rows = self.chunk_store.get_all_id_to_rows()
        
        # 2. Thực hiện OpenIE extraction
        logger.info("Performing OpenIE extraction...")
        ner_results, triple_results = self._perform_openie(documents, chunk_to_rows)
        
        # 3. Extract entities và facts
        logger.info("Extracting entities and facts...")
        entities, facts = self._extract_entities_and_facts(chunk_to_rows, ner_results, triple_results)
        
        # 4. Index entities và facts
        logger.info("Indexing entities...")
        self.entity_store.insert_strings(entities)
        
        logger.info("Indexing facts...")
        fact_strings = [str(fact) for fact in facts]
        self.fact_store.insert_strings(fact_strings)
        
        # 5. Build knowledge graph
        logger.info("Building knowledge graph...")
        self._build_knowledge_graph(chunk_to_rows, ner_results, triple_results)
        
        # 6. Save everything
        logger.info("Saving results...")
        self.knowledge_graph.save_graph()
        self.openie_manager.save_results()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Create result
        result = IndexingResult(
            num_docs_indexed=len(documents),
            num_entities_extracted=len(entities),
            num_triples_extracted=len(facts),
            processing_time=processing_time,
            graph_info=self.knowledge_graph.get_graph_info()
        )
        
        logger.info(f"Indexing completed in {processing_time:.2f}s")
        logger.info(f"Indexed {result.num_docs_indexed} documents, "
                   f"{result.num_entities_extracted} entities, "
                   f"{result.num_triples_extracted} facts")
        
        return result
    
    def _perform_openie(self, documents: List[str], chunk_to_rows: Dict[str, dict]) -> Tuple[Dict, Dict]:
        """Thực hiện OpenIE extraction"""
        # Check existing results
        existing_chunk_keys = self.openie_manager.get_existing_chunk_keys()
        
        # Find chunks that need processing
        chunks_to_process = {}
        for chunk_key, row in chunk_to_rows.items():
            if chunk_key not in existing_chunk_keys:
                chunks_to_process[chunk_key] = row
        
        if not chunks_to_process:
            logger.info("All chunks already processed, loading existing results")
            all_results = self.openie_manager.get_all_results()
            return reformat_openie_results(all_results)
        
        logger.info(f"Processing {len(chunks_to_process)} new chunks with OpenIE")
        
        if self.openie_model is None:
            logger.warning("No OpenIE model available, using empty results")
            # Create empty results for compatibility
            ner_results = {}
            triple_results = {}
            
            for chunk_key in chunks_to_process.keys():
                from .typing import NerRawOutput, TripleRawOutput
                ner_results[chunk_key] = NerRawOutput(
                    chunk_id=chunk_key, response=None, metadata={}, unique_entities=[]
                )
                triple_results[chunk_key] = TripleRawOutput(
                    chunk_id=chunk_key, response=None, metadata={}, triples=[]
                )
        else:
            # Perform actual OpenIE
            try:
                ner_results, triple_results = self.openie_model.batch_openie(chunks_to_process)
            except Exception as e:
                logger.error(f"Error in OpenIE processing: {e}")
                # Create empty results as fallback
                ner_results = {}
                triple_results = {}
                
                for chunk_key in chunks_to_process.keys():
                    from .typing import NerRawOutput, TripleRawOutput
                    ner_results[chunk_key] = NerRawOutput(
                        chunk_id=chunk_key, response=None, metadata={}, unique_entities=[]
                    )
                    triple_results[chunk_key] = TripleRawOutput(
                        chunk_id=chunk_key, response=None, metadata={}, triples=[]
                    )
        
        # Merge với existing results
        self.openie_manager.merge_results(chunks_to_process, ner_results, triple_results)
        
        # Return all results (existing + new)
        all_results = self.openie_manager.get_all_results()
        return reformat_openie_results(all_results)
    
    def _extract_entities_and_facts(self, chunk_to_rows: Dict[str, dict], 
                                   ner_results: Dict, triple_results: Dict) -> Tuple[List[str], List[tuple]]:
        """Extract entities và facts từ OpenIE results"""
        
        # Extract entities from all chunks
        chunk_ids = list(chunk_to_rows.keys())
        chunk_triples = []
        
        for chunk_id in chunk_ids:
            if chunk_id in triple_results:
                processed_triples = [text_processing(t) for t in triple_results[chunk_id].triples]
                chunk_triples.append(processed_triples)
            else:
                chunk_triples.append([])
        
        # Extract entity nodes
        entity_nodes, _ = extract_entity_nodes(chunk_triples)
        
        # Flatten facts
        facts = flatten_facts(chunk_triples)
        
        return entity_nodes, facts
    
    def _build_knowledge_graph(self, chunk_to_rows: Dict[str, dict], 
                              ner_results: Dict, triple_results: Dict):
        """Xây dựng knowledge graph"""
        
        chunk_ids = list(chunk_to_rows.keys())
        
        # Process triples
        chunk_triples = []
        chunk_triple_entities = []
        
        for chunk_id in chunk_ids:
            if chunk_id in triple_results:
                processed_triples = [text_processing(t) for t in triple_results[chunk_id].triples]
                chunk_triples.append(processed_triples)
                
                # Extract entities for this chunk
                entities = set()
                for triple in processed_triples:
                    if len(triple) >= 3:
                        entities.add(str(triple[0]))
                        entities.add(str(triple[2]))
                chunk_triple_entities.append(list(entities))
            else:
                chunk_triples.append([])
                chunk_triple_entities.append([])
        
        # Add fact edges
        self.knowledge_graph.add_fact_edges(chunk_ids, chunk_triples)
        
        # Add passage edges
        num_new_chunks = self.knowledge_graph.add_passage_edges(chunk_ids, chunk_triple_entities)
        
        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to add to graph")
            
            # Add synonymy edges if enabled
            synonymy_config = {
                'topk': self.config.synonymy_edge_topk,
                'sim_threshold': self.config.synonymy_edge_sim_threshold,
                'query_batch_size': self.config.synonymy_edge_query_batch_size,
                'key_batch_size': self.config.synonymy_edge_key_batch_size
            }
            
            if self.embedding_model:
                self.knowledge_graph.add_synonymy_edges(self.entity_store, synonymy_config)
            
            # Build final graph structure
            self.knowledge_graph.build_graph_structure(self.entity_store, self.chunk_store)
    
    def delete_documents(self, documents_to_delete: List[str]) -> bool:
        """
        Xóa documents khỏi index
        
        Args:
            documents_to_delete: List documents cần xóa
            
        Returns:
            True nếu thành công
        """
        try:
            # Get current documents
            current_docs = set(self.chunk_store.get_all_texts())
            docs_to_delete = [doc for doc in documents_to_delete if doc in current_docs]
            
            if not docs_to_delete:
                logger.info("No documents to delete")
                return True
            
            logger.info(f"Deleting {len(docs_to_delete)} documents")
            
            # Get chunk IDs to delete
            chunk_ids_to_delete = set()
            for doc in docs_to_delete:
                chunk_id = self.chunk_store.text_to_hash_id.get(doc)
                if chunk_id:
                    chunk_ids_to_delete.add(chunk_id)
            
            # Find entities and facts to delete
            entities_to_delete = set()
            facts_to_delete = set()
            
            # Update OpenIE results
            chunk_ids_to_keep = set(self.chunk_store.get_all_ids()) - chunk_ids_to_delete
            self.openie_manager.filter_results(chunk_ids_to_keep)
            
            # Delete from stores
            self.chunk_store.delete(list(chunk_ids_to_delete))
            
            # Delete from graph
            self.knowledge_graph.delete_vertices(list(chunk_ids_to_delete))
            
            # Save changes
            self.knowledge_graph.save_graph()
            self.openie_manager.save_results()
            
            logger.info("Document deletion completed")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_embedding_stores(self) -> Dict[str, EmbeddingStore]:
        """Lấy embedding stores"""
        return {
            'chunk_store': self.chunk_store,
            'entity_store': self.entity_store,
            'fact_store': self.fact_store
        }
    
    def get_knowledge_graph(self) -> KnowledgeGraph:
        """Lấy knowledge graph"""
        return self.knowledge_graph
    
    def is_ready_for_retrieval(self) -> bool:
        """Kiểm tra có sẵn sàng cho retrieval không"""
        return (self.knowledge_graph.is_ready() and 
                len(self.chunk_store.get_all_ids()) > 0 and
                len(self.entity_store.get_all_ids()) > 0)
