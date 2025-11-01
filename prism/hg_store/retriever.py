import os
from tqdm import tqdm
from typing import Optional, List
import numpy as np

from utils.logs import logger
from utils.constants import ROOT_PATH

from prism.settings import hg_store_name, prism_embedding_model
from prism.hg_store.store import EmbeddingStore
from prism.hg_store.information_extractor import InformationExtractor

class Retriever:
    def __init__(self):
        self.embedding_model = prism_embedding_model
        self.store = EmbeddingStore(
            self.embedding_model,
            os.path.join(ROOT_PATH, "cache", "hg_store"),
            32,
            hg_store_name
        )
        self.information_extractor = InformationExtractor()
        
    async def retrieve(self, query: str, task_type: str, top_k: int = 10):
        """
        Retrieve top-k most similar documents based on query embedding with task-specific scoring
        
        Args:
            query (str): Search query text
            task_type (str): Task type for filtering documents
            top_k (int): Number of top results to return (default: 10)
            
        Returns:
            List[dict]: Top-k results sorted by similarity score (descending)
        """
        # Extract information from query
        ner_query = await self.information_extractor.ner_query(query)
        classification_content = f"Datasets: {', '.join(ner_query.get('datasets', ''))}. Classes: {', '.join(ner_query.get('classes', ''))}. Number of classes: {ner_query.get('num_classes', '')}"
        tabular_content = f"Input format: {ner_query.get('input_format', '')}"
        
        # Generate hypothesis query
        query_document_hypothesis = await self.information_extractor.description_hypothesis(query, ner_query)
        
        # Get embeddings for different query types
        query_hypothesis_embedding = await self.embedding_model.encode(query_document_hypothesis)
        
        # Get candidate documents
        rows = self.store.get_rows_by_task_type([task_type])
        
        if not rows:
            logger.warning("No documents found for retrieval")
            return []
            
        # Convert rows to list format for indexing
        rows_list = list(rows.values()) if isinstance(rows, dict) else rows
        
        # Extract embeddings from documents
        document_embeddings = np.array([row["embedding"] for row in rows_list])
        classification_embeddings = np.array([row["classification_embedding"] for row in rows_list])
        tabular_embeddings = np.array([row["tabular_embedding"] for row in rows_list])
        
        # Ensure query embedding is 1D
        if query_hypothesis_embedding.ndim > 1:
            query_hypothesis_embedding = query_hypothesis_embedding.flatten()
            
        # Handle zero norm cases
        query_norm = np.linalg.norm(query_hypothesis_embedding)
        if query_norm == 0:
            logger.warning("Query embedding has zero norm")
            return []
            
        # Calculate similarity scores based on task type
        similarity_scores = await self._calculate_task_specific_scores(
            task_type, 
            query_hypothesis_embedding,
            query_norm,
            document_embeddings,
            classification_embeddings,
            tabular_embeddings,
            classification_content,
            tabular_content
        )
        
        # Get top-k indices sorted by similarity (descending order)
        top_k = min(top_k, len(similarity_scores))
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            result = {
                "score": float(similarity_scores[idx]),
                "document": rows_list[idx],
                "hash_id": rows_list[idx]["hash_id"],
                "content": rows_list[idx]["content"],
                "task_type": rows_list[idx].get("task_type", "unknown"),
                "metadata": rows_list[idx].get("metadata", {})
            }
            results.append(result)
            
        logger.info(f"Retrieved top-{len(results)} results for query: '{query[:50]}...' with task_type: {task_type}")
        return results
    
    async def _calculate_task_specific_scores(self, task_type: str, query_embedding: np.ndarray, query_norm: float,
                                            document_embeddings: np.ndarray, classification_embeddings: np.ndarray, 
                                            tabular_embeddings: np.ndarray, classification_content: str, tabular_content: str):
        """
        Calculate similarity scores based on task type with different weighting strategies
        """
        # Normalize main query embedding
        query_embedding_normalized = query_embedding / query_norm
        
        # Calculate main embedding similarity
        doc_norms = np.linalg.norm(document_embeddings, axis=1)
        document_embeddings_normalized = document_embeddings / doc_norms[:, np.newaxis]
        main_scores = np.dot(document_embeddings_normalized, query_embedding_normalized)
        
        # For classification tasks: 0.2 * classification_score + 0.8 * main_score
        if task_type in ["text-classification", "image-classification", "video-classification", "audio-classification"]:
            # Get classification query embedding
            classification_query_embedding = await self.embedding_model.encode(classification_content)
            if classification_query_embedding.ndim > 1:
                classification_query_embedding = classification_query_embedding.flatten()
                
            # Calculate classification similarity
            classification_query_norm = np.linalg.norm(classification_query_embedding)
            if classification_query_norm > 0:
                classification_query_normalized = classification_query_embedding / classification_query_norm
                classification_doc_norms = np.linalg.norm(classification_embeddings, axis=1)
                classification_embeddings_normalized = classification_embeddings / classification_doc_norms[:, np.newaxis]
                classification_scores = np.dot(classification_embeddings_normalized, classification_query_normalized)
                
                # Weighted combination: 0.2 * classification + 0.8 * main
                final_scores = 0.1 * classification_scores + 0.9 * main_scores
            else:
                final_scores = main_scores
                
        # For tabular tasks: 0.2 * tabular_score + 0.8 * main_score
        elif task_type in ["tabular-classification", "tabular-regression"]:
            # Get tabular query embedding
            tabular_query_embedding = await self.embedding_model.encode(tabular_content)
            if tabular_query_embedding.ndim > 1:
                tabular_query_embedding = tabular_query_embedding.flatten()
                
            # Calculate tabular similarity
            tabular_query_norm = np.linalg.norm(tabular_query_embedding)
            if tabular_query_norm > 0:
                tabular_query_normalized = tabular_query_embedding / tabular_query_norm
                tabular_doc_norms = np.linalg.norm(tabular_embeddings, axis=1)
                tabular_embeddings_normalized = tabular_embeddings / tabular_doc_norms[:, np.newaxis]
                tabular_scores = np.dot(tabular_embeddings_normalized, tabular_query_normalized)
                
                # Weighted combination: 0.2 * tabular + 0.8 * main
                final_scores = 0.2 * tabular_scores + 0.8 * main_scores
            else:
                final_scores = main_scores
                
        # For other tasks: pure main embedding similarity
        else:
            final_scores = main_scores
            
        return final_scores