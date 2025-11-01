import os
import logging
import time
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import igraph as ig
from collections import defaultdict
import re
from tqdm import tqdm

from .typing import Triple, GraphInfo
from .utils import compute_mdhash_id, text_processing, extract_entity_nodes, flatten_facts, min_max_normalize, retrieve_knn

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """Quản lý Knowledge Graph cho HippoRAG"""
    
    def __init__(self, working_dir: str, is_directed: bool = False, force_rebuild: bool = False):
        self.working_dir = working_dir
        self.is_directed = is_directed
        self.force_rebuild = force_rebuild
        
        # Graph structure
        self.graph: ig.Graph = None
        self.node_to_node_stats: Dict[Tuple[str, str], float] = {}
        self.ent_node_to_chunk_ids: Dict[str, Set[str]] = {}
        
        # Node mappings
        self.node_name_to_vertex_idx: Dict[str, int] = {}
        self.entity_node_keys: List[str] = []
        self.passage_node_keys: List[str] = []
        self.entity_node_idxs: List[int] = []
        self.passage_node_idxs: List[int] = []
        
        # Paths
        self._graph_pickle_filename = os.path.join(working_dir, "graph.pickle")
        
        # Initialize graph
        self._initialize_graph()
    
    def _initialize_graph(self):
        """Khởi tạo hoặc load graph từ file"""
        preloaded_graph = None
        
        if not self.force_rebuild and os.path.exists(self._graph_pickle_filename):
            try:
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)
                logger.info(f"Loaded graph from {self._graph_pickle_filename} with "
                          f"{preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges")
            except Exception as e:
                logger.warning(f"Failed to load graph: {e}")
        
        if preloaded_graph is None:
            self.graph = ig.Graph(directed=self.is_directed)
            logger.info("Initialized new empty graph")
        else:
            self.graph = preloaded_graph
    
    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[List[Triple]]):
        """Thêm fact edges từ triples vào graph"""
        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()
        
        logger.info("Adding OpenIE triples to graph.")
        
        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples), desc="Processing chunks"):
            entities_in_chunk = set()
            
            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    if len(triple) >= 3:
                        triple = tuple(triple)
                        
                        node_key = compute_mdhash_id(content=triple[0], prefix="entity-")
                        node_2_key = compute_mdhash_id(content=triple[2], prefix="entity-")
                        
                        # Update edge statistics
                        self.node_to_node_stats[(node_key, node_2_key)] = \
                            self.node_to_node_stats.get((node_key, node_2_key), 0.0) + 1
                        self.node_to_node_stats[(node_2_key, node_key)] = \
                            self.node_to_node_stats.get((node_2_key, node_key), 0.0) + 1
                        
                        entities_in_chunk.add(node_key)
                        entities_in_chunk.add(node_2_key)
                
                # Update entity to chunk mapping
                for node in entities_in_chunk:
                    if node not in self.ent_node_to_chunk_ids:
                        self.ent_node_to_chunk_ids[node] = set()
                    self.ent_node_to_chunk_ids[node].add(chunk_key)
    
    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]) -> int:
        """Thêm edges kết nối passage nodes với entity nodes"""
        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()
        
        num_new_chunks = 0
        
        logger.info("Connecting passage nodes to phrase nodes.")
        
        for idx, chunk_key in tqdm(enumerate(chunk_ids), desc="Adding passage edges"):
            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")
                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0
                
                num_new_chunks += 1
        
        return num_new_chunks
    
    def add_synonymy_edges(self, entity_embedding_store, synonymy_config: dict):
        """Thêm synonymy edges giữa các entities tương tự"""
        logger.info("Expanding graph with synonymy edges")
        
        entity_id_to_row = entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(entity_id_to_row.keys())
        
        if not entity_node_keys:
            logger.warning("No entity nodes found for synonymy edge creation")
            return
        
        logger.info(f"Performing KNN retrieval for {len(entity_node_keys)} phrase nodes.")
        
        entity_embs = entity_embedding_store.get_embeddings(entity_node_keys)
        
        if len(entity_embs) == 0:
            logger.warning("No embeddings found for entities")
            return
        
        # KNN search for synonyms
        query_node_key2knn_node_keys = retrieve_knn(
            query_ids=entity_node_keys,
            key_ids=entity_node_keys,
            query_vecs=entity_embs,
            key_vecs=entity_embs,
            k=synonymy_config.get('topk', 10),
            query_batch_size=synonymy_config.get('query_batch_size', 64),
            key_batch_size=synonymy_config.get('key_batch_size', 64)
        )
        
        num_synonym_triple = 0
        sim_threshold = synonymy_config.get('sim_threshold', 0.8)
        
        for node_key in tqdm(query_node_key2knn_node_keys.keys(), desc="Adding synonymy edges"):
            entity = entity_id_to_row[node_key]["content"]
            
            # Chỉ xử lý entities có ít nhất 3 ký tự alphanumeric
            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]
                
                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < sim_threshold or num_nns > 100:
                        break
                    
                    nn_phrase = entity_id_to_row[nn]["content"]
                    
                    if nn != node_key and nn_phrase != '':
                        self.node_to_node_stats[(node_key, nn)] = score
                        num_synonym_triple += 1
                        num_nns += 1
        
        logger.info(f"Added {num_synonym_triple} synonymy edges")
    
    def build_graph_structure(self, entity_embedding_store, chunk_embedding_store):
        """Xây dựng cấu trúc graph với nodes và edges"""
        self._add_nodes(entity_embedding_store, chunk_embedding_store)
        self._add_edges()
        self._update_node_mappings()
    
    def _add_nodes(self, entity_embedding_store, chunk_embedding_store):
        """Thêm nodes vào graph"""
        existing_nodes = {}
        if "name" in self.graph.vs.attribute_names():
            existing_nodes = {v["name"]: v for v in self.graph.vs}
        
        # Collect all nodes
        entity_to_row = entity_embedding_store.get_all_id_to_rows()
        passage_to_row = chunk_embedding_store.get_all_id_to_rows()
        
        node_to_rows = {}
        node_to_rows.update(entity_to_row)
        node_to_rows.update(passage_to_row)
        
        # Prepare new nodes
        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)
        
        # Add new nodes to graph
        if new_nodes:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)
            logger.info(f"Added {len(next(iter(new_nodes.values())))} new nodes to graph")
    
    def _add_edges(self):
        """Thêm edges vào graph từ node_to_node_stats"""
        if not self.node_to_node_stats:
            logger.warning("No edge statistics found")
            return
        
        # Prepare edges
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]:  # Skip self-loops
                continue
            
            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({"weight": weight})
        
        # Validate edges against current nodes
        valid_edges = []
        valid_weights = {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        
        for source_id, target_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_id in current_node_ids and target_id in current_node_ids:
                valid_edges.append((source_id, target_id))
                valid_weights["weight"].append(edge_d.get("weight", 1.0))
            else:
                logger.debug(f"Edge {source_id} -> {target_id} is not valid.")
        
        # Add edges to graph
        if valid_edges:
            self.graph.add_edges(valid_edges, attributes=valid_weights)
            logger.info(f"Added {len(valid_edges)} edges to graph")
    
    def _update_node_mappings(self):
        """Cập nhật mappings giữa node names và vertex indices"""
        self.node_name_to_vertex_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
        
        # Update node lists and indices
        self.entity_node_keys = [name for name in self.node_name_to_vertex_idx.keys() 
                                if name.startswith("entity-")]
        self.passage_node_keys = [name for name in self.node_name_to_vertex_idx.keys() 
                                 if name.startswith("chunk-")]
        
        self.entity_node_idxs = [self.node_name_to_vertex_idx[key] for key in self.entity_node_keys]
        self.passage_node_idxs = [self.node_name_to_vertex_idx[key] for key in self.passage_node_keys]
    
    def run_personalized_pagerank(self, reset_prob: np.ndarray, damping: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Chạy Personalized PageRank trên graph"""
        if damping is None:
            damping = 0.5
        
        # Clean reset probabilities
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        
        try:
            pagerank_scores = self.graph.personalized_pagerank(
                vertices=range(len(self.node_name_to_vertex_idx)),
                damping=damping,
                directed=False,
                weights='weight',
                reset=reset_prob,
                implementation='prpack'
            )
            
            # Extract scores for document nodes only
            doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
            sorted_doc_ids = np.argsort(doc_scores)[::-1]
            sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]
            
            return sorted_doc_ids, sorted_doc_scores
            
        except Exception as e:
            logger.error(f"Error running PageRank: {e}")
            # Return empty results
            return np.array([]), np.array([])
    
    def delete_vertices(self, vertex_ids: List[str]):
        """Xóa vertices khỏi graph"""
        if not vertex_ids:
            return
        
        # Convert names to indices
        indices_to_delete = []
        for vertex_id in vertex_ids:
            if vertex_id in self.node_name_to_vertex_idx:
                indices_to_delete.append(self.node_name_to_vertex_idx[vertex_id])
        
        if indices_to_delete:
            self.graph.delete_vertices(indices_to_delete)
            self._update_node_mappings()
            logger.info(f"Deleted {len(indices_to_delete)} vertices from graph")
    
    def save_graph(self):
        """Lưu graph vào file"""
        os.makedirs(os.path.dirname(self._graph_pickle_filename), exist_ok=True)
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Graph saved to {self._graph_pickle_filename} with "
                   f"{self.graph.vcount()} nodes, {self.graph.ecount()} edges")
    
    def get_graph_info(self) -> GraphInfo:
        """Lấy thông tin thống kê về graph"""
        num_phrase_nodes = len(self.entity_node_keys)
        num_passage_nodes = len(self.passage_node_keys)
        num_total_nodes = num_phrase_nodes + num_passage_nodes
        
        # Count different types of triples
        passage_nodes_set = set(self.passage_node_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        
        # Estimate extracted triples (those not involving passage nodes and not synonymy)
        num_extracted_triples = max(0, len(self.node_to_node_stats) - num_triples_with_passage_node)
        num_synonymy_triples = 0  # Will be updated if synonymy detection is implemented
        
        return GraphInfo(
            num_phrase_nodes=num_phrase_nodes,
            num_passage_nodes=num_passage_nodes,
            num_total_nodes=num_total_nodes,
            num_extracted_triples=num_extracted_triples,
            num_triples_with_passage_node=num_triples_with_passage_node,
            num_synonymy_triples=num_synonymy_triples,
            num_total_triples=len(self.node_to_node_stats)
        )
    
    def is_ready(self) -> bool:
        """Kiểm tra graph đã sẵn sàng cho retrieval chưa"""
        return (self.graph is not None and 
                self.graph.vcount() > 0 and 
                len(self.node_name_to_vertex_idx) > 0)
    
    def get_node_weights_for_ppr(self, phrase_weights: np.ndarray, passage_weights: np.ndarray) -> np.ndarray:
        """Combine phrase và passage weights cho PPR"""
        node_weights = np.zeros(self.graph.vcount())
        
        # Set phrase weights
        for i, entity_idx in enumerate(self.entity_node_idxs):
            if i < len(phrase_weights):
                node_weights[entity_idx] = phrase_weights[i]
        
        # Set passage weights
        for i, passage_idx in enumerate(self.passage_node_idxs):
            if i < len(passage_weights):
                node_weights[passage_idx] = passage_weights[i]
        
        return node_weights
