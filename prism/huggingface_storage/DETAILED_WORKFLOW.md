# HippoRAG Modular Implementation - Chi tiết Workflow

## 📋 Tổng quan

HippoRAG là một hệ thống Retrieval-Augmented Generation (RAG) sử dụng knowledge graph để cải thiện khả năng retrieval. Phiên bản modular này chia nhỏ quá trình thành các bước rõ ràng, dễ hiểu và maintain.

## 🔄 Workflow tổng thể

```
Documents → Indexing → Knowledge Graph → Retrieval → QA → Answers
    ↓           ↓              ↓              ↓         ↓
   Text    OpenIE+Emb     Graph Build    Fact+PPR   Answer
```

---

## 🏗️ PHASE 1: INITIALIZATION & SETUP

### 1.1 Khởi tạo Configuration

```python
config = RetrievalConfig(
    retrieval_top_k=10,      # Số documents trả về
    linking_top_k=5,         # Số facts top để link
    qa_top_k=3,              # Số docs dùng cho QA
    passage_node_weight=0.05, # Trọng số passage trong PPR
    damping=0.5,             # Damping factor cho PageRank
    synonymy_edge_topk=10,   # Top synonyms
    synonymy_edge_sim_threshold=0.8,  # Ngưỡng similarity
    embedding_batch_size=32   # Batch size cho embedding
)
```

**Chi tiết:**
- `RetrievalConfig` được định nghĩa trong `typing.py`
- Lưu trữ tất cả hyperparameters cần thiết
- Sử dụng dataclass để type safety
- Có thể override từng parameter riêng lẻ

### 1.2 Khởi tạo Models

```python
# Cần 3 models chính:
embedding_model = YourEmbeddingModel()  # Để encode text thành vectors
openie_model = YourOpenIEModel()        # Để extract entities và relations
llm_model = YourLLMModel()              # Để generate answers
```

**Chi tiết:**
- **Embedding Model**: Chuyển text thành dense vectors (thường 768/1024 dims)
- **OpenIE Model**: Extract (subject, predicate, object) triples từ text
- **LLM Model**: Generate câu trả lời từ retrieved documents

---

## 🗂️ PHASE 2: INDEXING WORKFLOW

### 2.1 Khởi tạo HippoIndexer

```python
indexer = HippoIndexer(
    working_dir="./storage",
    openie_model=openie_model,
    embedding_model=embedding_model,
    config=config,
    force_rebuild=False
)
```

**Chi tiết trong `indexing.py`:**

1. **Tạo working directory** nếu chưa tồn tại
2. **Initialize 3 EmbeddingStores:**
   - `chunk_store`: Lưu documents và embeddings
   - `entity_store`: Lưu entities và embeddings  
   - `fact_store`: Lưu facts (triples) và embeddings
3. **Initialize KnowledgeGraph** với igraph
4. **Initialize OpenIEResultsManager** để lưu OpenIE results

#### 2.1.1 EmbeddingStore Details

Mỗi `EmbeddingStore` quản lý:
```python
class EmbeddingStore:
    def __init__(self, embedding_model, storage_path, batch_size, store_type):
        self.embedding_model = embedding_model
        self.storage_path = storage_path          # Thư mục lưu data
        self.store_type = store_type              # "chunk", "entity", "fact"
        self.text_to_hash_id = {}                 # Mapping text -> hash_id
        self._id_to_rows = {}                     # Mapping hash_id -> data
```

**Cấu trúc file storage:**
```
storage/
├── chunk_embeddings/
│   ├── metadata.json        # Text mappings
│   └── embeddings.npy       # Numpy array vectors
├── entity_embeddings/
│   ├── metadata.json
│   └── embeddings.npy
└── fact_embeddings/
    ├── metadata.json
    └── embeddings.npy
```

### 2.2 Document Processing Pipeline

```python
result = indexer.index_documents(documents)
```

#### 2.2.1 Bước 1: Insert Documents vào Chunk Store

**Location:** `indexing.py`, method `index_documents()`

```python
# 1. Insert documents vào chunk store
self.chunk_store.insert_strings(documents)
chunk_to_rows = self.chunk_store.get_all_id_to_rows()
```

**Chi tiết process:**

1. **Compute Hash IDs:**
   ```python
   for document in documents:
       hash_id = compute_mdhash_id(document, prefix="chunk-")
       # hash_id = "chunk-" + md5(document)
   ```

2. **Check Duplicates:**
   - Chỉ process documents chưa có trong store
   - Skip documents đã được index

3. **Generate Embeddings:**
   ```python
   if new_documents:
       embeddings = embedding_model.batch_encode(new_documents, norm=True)
       # embeddings shape: (n_docs, embedding_dim)
   ```

4. **Save to Storage:**
   ```python
   # Lưu vào metadata.json
   metadata = {
       'text_to_hash_id': {doc: hash_id, ...},
       'id_to_rows': {hash_id: {'hash_id': hash_id, 'content': doc}, ...}
   }
   
   # Lưu embeddings vào .npy file
   np.save(embedding_path, embeddings)
   ```

#### 2.2.2 Bước 2: OpenIE Extraction

**Location:** `indexing.py`, method `_perform_openie()`

```python
ner_results, triple_results = self._perform_openie(documents, chunk_to_rows)
```

**Chi tiết process:**

1. **Check Existing Results:**
   ```python
   existing_chunk_keys = self.openie_manager.get_existing_chunk_keys()
   chunks_to_process = {k: v for k, v in chunk_to_rows.items() 
                       if k not in existing_chunk_keys}
   ```

2. **Perform OpenIE:**
   ```python
   # Gọi OpenIE model để extract entities và relations
   ner_results, triple_results = self.openie_model.batch_openie(chunks_to_process)
   ```

   **Output format:**
   ```python
   ner_results = {
       "chunk-abc123": NerRawOutput(
           chunk_id="chunk-abc123",
           response="Raw model response",
           metadata={"confidence": 0.9},
           unique_entities=["AI", "machine learning", "algorithms"]
       )
   }
   
   triple_results = {
       "chunk-abc123": TripleRawOutput(
           chunk_id="chunk-abc123", 
           response="Raw model response",
           metadata={"confidence": 0.8},
           triples=[
               ("AI", "transforms", "industry"),
               ("machine learning", "uses", "algorithms")
           ]
       )
   }
   ```

3. **Merge với Existing Results:**
   ```python
   self.openie_manager.merge_results(chunks_to_process, ner_results, triple_results)
   ```

4. **Save OpenIE Results:**
   ```python
   # Lưu vào openie_results.json
   {
       "docs": [
           {
               "idx": "chunk-abc123",
               "passage": "AI transforms industry...",
               "extracted_entities": ["AI", "machine learning"],
               "extracted_triples": [("AI", "transforms", "industry")]
           }
       ],
       "avg_ent_chars": 12.5,
       "avg_ent_words": 2.1
   }
   ```

#### 2.2.3 Bước 3: Extract Entities và Facts

**Location:** `indexing.py`, method `_extract_entities_and_facts()`

```python
entities, facts = self._extract_entities_and_facts(chunk_to_rows, ner_results, triple_results)
```

**Chi tiết process:**

1. **Process Triples:**
   ```python
   chunk_triples = []
   for chunk_id in chunk_ids:
       if chunk_id in triple_results:
           processed_triples = [text_processing(t) for t in triple_results[chunk_id].triples]
           chunk_triples.append(processed_triples)
   ```

2. **Extract Entity Nodes:**
   ```python
   def extract_entity_nodes(chunk_triples):
       all_entities = set()
       for chunk_triples_list in chunk_triples:
           for triple in chunk_triples_list:
               subject = str(triple[0]).strip().lower()
               obj = str(triple[2]).strip().lower()
               all_entities.add(subject)
               all_entities.add(obj)
       return list(all_entities)
   ```

3. **Flatten Facts:**
   ```python
   def flatten_facts(chunk_triples):
       flattened = []
       for chunk_triples_list in chunk_triples:
           for triple in chunk_triples_list:
               if len(triple) >= 3:
                   flattened.append(tuple(triple[:3]))
       return flattened
   ```

**Output:**
```python
entities = ["ai", "machine learning", "algorithms", "industry", ...]
facts = [
    ("ai", "transforms", "industry"),
    ("machine learning", "uses", "algorithms"),
    ...
]
```

#### 2.2.4 Bước 4: Index Entities và Facts

```python
# Index entities
self.entity_store.insert_strings(entities)

# Index facts
fact_strings = [str(fact) for fact in facts]  # Convert tuples to strings
self.fact_store.insert_strings(fact_strings)
```

**Chi tiết:**
- Entities được embed riêng với instruction đặc biệt cho entities
- Facts được convert thành string format: `"('ai', 'transforms', 'industry')"`
- Mỗi loại có embedding riêng để optimize cho use case

#### 2.2.5 Bước 5: Build Knowledge Graph

**Location:** `indexing.py`, method `_build_knowledge_graph()`

```python
self._build_knowledge_graph(chunk_to_rows, ner_results, triple_results)
```

**Chi tiết process:**

1. **Prepare Data:**
   ```python
   chunk_ids = list(chunk_to_rows.keys())
   chunk_triples = []          # Triples cho mỗi chunk
   chunk_triple_entities = []  # Entities cho mỗi chunk
   
   for chunk_id in chunk_ids:
       if chunk_id in triple_results:
           processed_triples = [text_processing(t) for t in triple_results[chunk_id].triples]
           chunk_triples.append(processed_triples)
           
           # Extract entities cho chunk này
           entities = set()
           for triple in processed_triples:
               if len(triple) >= 3:
                   entities.add(str(triple[0]))
                   entities.add(str(triple[2]))
           chunk_triple_entities.append(list(entities))
   ```

2. **Add Fact Edges:**
   ```python
   self.knowledge_graph.add_fact_edges(chunk_ids, chunk_triples)
   ```

   **Chi tiết trong `graph.py`:**
   ```python
   def add_fact_edges(self, chunk_ids, chunk_triples):
       for chunk_key, triples in zip(chunk_ids, chunk_triples):
           entities_in_chunk = set()
           
           for triple in triples:
               if len(triple) >= 3:
                   # Tạo hash IDs cho subject và object
                   node_key = compute_mdhash_id(content=triple[0], prefix="entity-")
                   node_2_key = compute_mdhash_id(content=triple[2], prefix="entity-")
                   
                   # Cập nhật edge statistics (bidirectional)
                   self.node_to_node_stats[(node_key, node_2_key)] += 1
                   self.node_to_node_stats[(node_2_key, node_key)] += 1
                   
                   entities_in_chunk.add(node_key)
                   entities_in_chunk.add(node_2_key)
           
           # Map entities tới chunks chứa chúng
           for node in entities_in_chunk:
               self.ent_node_to_chunk_ids[node].add(chunk_key)
   ```

3. **Add Passage Edges:**
   ```python
   num_new_chunks = self.knowledge_graph.add_passage_edges(chunk_ids, chunk_triple_entities)
   ```

   **Chi tiết:**
   ```python
   def add_passage_edges(self, chunk_ids, chunk_triple_entities):
       for idx, chunk_key in enumerate(chunk_ids):
           for chunk_ent in chunk_triple_entities[idx]:
               node_key = compute_mdhash_id(chunk_ent, prefix="entity-")
               # Connect passage node với entity node
               self.node_to_node_stats[(chunk_key, node_key)] = 1.0
   ```

4. **Add Synonymy Edges (nếu có embedding model):**
   ```python
   if self.embedding_model:
       synonymy_config = {
           'topk': self.config.synonymy_edge_topk,
           'sim_threshold': self.config.synonymy_edge_sim_threshold,
           # ...
       }
       self.knowledge_graph.add_synonymy_edges(self.entity_store, synonymy_config)
   ```

   **Chi tiết trong `graph.py`:**
   ```python
   def add_synonymy_edges(self, entity_embedding_store, synonymy_config):
       # Get all entity embeddings
       entity_node_keys = list(entity_id_to_row.keys())
       entity_embs = entity_embedding_store.get_embeddings(entity_node_keys)
       
       # KNN search để tìm similar entities
       query_node_key2knn_node_keys = retrieve_knn(
           query_ids=entity_node_keys,
           key_ids=entity_node_keys,
           query_vecs=entity_embs,
           key_vecs=entity_embs,
           k=synonymy_config['topk']
       )
       
       # Add synonymy edges
       for node_key in entity_node_keys:
           entity = entity_id_to_row[node_key]["content"]
           
           # Chỉ process entities có ít nhất 3 ký tự alphanumeric
           if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
               nns = query_node_key2knn_node_keys[node_key]
               
               for nn, score in zip(nns[0], nns[1]):
                   if score < sim_threshold:
                       break
                   
                   if nn != node_key:
                       self.node_to_node_stats[(node_key, nn)] = score
   ```

5. **Build Final Graph Structure:**
   ```python
   self.knowledge_graph.build_graph_structure(self.entity_store, self.chunk_store)
   ```

   **Chi tiết:**
   - Add all nodes vào igraph
   - Add all edges với weights
   - Create mappings từ node names → vertex indices
   - Update entity_node_keys, passage_node_keys, etc.

#### 2.2.6 Bước 6: Save Everything

```python
self.knowledge_graph.save_graph()        # Lưu graph.pickle
self.openie_manager.save_results()       # Lưu openie_results.json
```

**Chi tiết file outputs:**
```
storage/
├── chunk_embeddings/
│   ├── metadata.json
│   └── embeddings.npy
├── entity_embeddings/
│   ├── metadata.json  
│   └── embeddings.npy
├── fact_embeddings/
│   ├── metadata.json
│   └── embeddings.npy
├── graph.pickle              # iGraph object
└── openie_results.json       # OpenIE extraction results
```

**IndexingResult:**
```python
result = IndexingResult(
    num_docs_indexed=len(documents),
    num_entities_extracted=len(entities),
    num_triples_extracted=len(facts),
    processing_time=end_time - start_time,
    graph_info=self.knowledge_graph.get_graph_info()
)
```

---

## 🔍 PHASE 3: RETRIEVAL WORKFLOW

### 3.1 Khởi tạo HippoRetriever

```python
embedding_stores = indexer.get_embedding_stores()
knowledge_graph = indexer.get_knowledge_graph()

retriever = HippoRetriever(
    knowledge_graph=knowledge_graph,
    embedding_stores=embedding_stores,
    llm_model=llm_model,
    rerank_filter=None,
    config=config
)
```

**Chi tiết trong `retrieval.py`:**

1. **Store References:**
   ```python
   self.chunk_store = embedding_stores.get('chunk_store')
   self.entity_store = embedding_stores.get('entity_store') 
   self.fact_store = embedding_stores.get('fact_store')
   ```

2. **Initialize Caches:**
   ```python
   self.query_embeddings_cache = {
       'triple': {},    # Cache cho fact retrieval embeddings
       'passage': {}    # Cache cho passage retrieval embeddings
   }
   ```

3. **Prepare Retrieval Data:**
   ```python
   def _prepare_retrieval_data(self):
       # Load precomputed embeddings cho fast retrieval
       entity_keys = self.knowledge_graph.entity_node_keys
       self.entity_embeddings = self.entity_store.get_embeddings(entity_keys)
       
       passage_keys = self.knowledge_graph.passage_node_keys  
       self.passage_embeddings = self.chunk_store.get_embeddings(passage_keys)
       
       fact_keys = self.fact_store.get_all_ids()
       self.fact_embeddings = self.fact_store.get_embeddings(fact_keys)
   ```

### 3.2 Query Processing Pipeline

```python
results = retriever.retrieve(queries, embedding_model)
```

#### 3.2.1 Bước 1: Get Query Embeddings

**Location:** `retrieval.py`, method `get_query_embeddings()`

```python
def get_query_embeddings(self, queries, embedding_model):
    new_queries = []
    
    for query in queries:
        if query not in self.query_embeddings_cache['triple'] or \
           query not in self.query_embeddings_cache['passage']:
            new_queries.append(query)
    
    if new_queries:
        # Embeddings cho fact retrieval
        fact_embeddings = embedding_model.batch_encode(
            new_queries, 
            instruction="query_to_fact",
            norm=True
        )
        
        # Embeddings cho passage retrieval  
        passage_embeddings = embedding_model.batch_encode(
            new_queries,
            instruction="query_to_passage", 
            norm=True
        )
        
        # Cache embeddings
        for query, fact_emb, pass_emb in zip(new_queries, fact_embeddings, passage_embeddings):
            self.query_embeddings_cache['triple'][query] = fact_emb
            self.query_embeddings_cache['passage'][query] = pass_emb
```

**Chi tiết:**
- Tạo 2 loại embeddings khác nhau cho mỗi query
- `query_to_fact`: Optimize để match với facts/triples
- `query_to_passage`: Optimize để match với passages
- Cache để tránh recompute cho queries giống nhau

#### 3.2.2 Bước 2: Fact Scoring và Retrieval

**Cho mỗi query:**

1. **Get Fact Scores:**
   ```python
   query_fact_scores = self.get_fact_scores(query)
   ```

   **Chi tiết:**
   ```python
   def get_fact_scores(self, query):
       query_embedding = self.query_embeddings_cache['triple'].get(query)
       
       # Compute similarity với tất cả facts
       query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T)
       query_fact_scores = np.squeeze(query_fact_scores)
       
       # Normalize scores về [0,1]
       query_fact_scores = min_max_normalize(query_fact_scores)
       return query_fact_scores
   ```

2. **Rerank Facts:**
   ```python
   top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
   ```

   **Chi tiết:**
   ```python
   def rerank_facts(self, query, query_fact_scores):
       link_top_k = self.config.linking_top_k
       
       # Get top facts by score
       if len(query_fact_scores) <= link_top_k:
           candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
       else:
           candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
       
       # Get actual facts
       fact_keys = self.fact_store.get_all_ids()
       candidate_fact_ids = [fact_keys[idx] for idx in candidate_fact_indices]
       fact_rows = self.fact_store.get_rows(candidate_fact_ids)
       candidate_facts = [eval(fact_rows[fid]['content']) for fid in candidate_fact_ids]
       
       # Apply reranking filter nếu có
       if self.rerank_filter:
           top_k_fact_indices, top_k_facts, _ = self.rerank_filter(
               query, candidate_facts, candidate_fact_indices, len_after_rerank=link_top_k
           )
       else:
           top_k_fact_indices = candidate_fact_indices
           top_k_facts = candidate_facts
       
       return top_k_fact_indices, top_k_facts, rerank_log
   ```

#### 3.2.3 Bước 3: Graph Search với Facts

```python
if len(top_k_facts) == 0:
    # Fallback về Dense Passage Retrieval
    sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
else:
    # Sử dụng facts để guide graph search
    sorted_doc_ids, sorted_doc_scores = self.graph_search_with_facts(
        query, top_k_facts, top_k_fact_indices, query_fact_scores
    )
```

**Chi tiết Graph Search trong `retrieval.py`:**

1. **Initialize Weights:**
   ```python
   phrase_weights = np.zeros(len(self.knowledge_graph.entity_node_keys))
   passage_weights = np.zeros(len(self.knowledge_graph.passage_node_keys))
   ```

2. **Compute Phrase Weights từ Facts:**
   ```python
   linking_score_map = {}
   phrase_scores = {}
   
   for rank, fact in enumerate(top_k_facts):
       if len(fact) >= 3:
           subject_phrase = fact[0].lower()
           object_phrase = fact[2].lower()
           fact_score = query_fact_scores[top_k_fact_indices[rank]]
           
           for phrase in [subject_phrase, object_phrase]:
               phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
               
               # Find phrase index trong entity_node_keys
               if phrase_key in self.knowledge_graph.entity_node_keys:
                   phrase_idx = self.knowledge_graph.entity_node_keys.index(phrase_key)
                   
                   # Weight by inverse frequency
                   ent_chunk_count = len(self.knowledge_graph.ent_node_to_chunk_ids.get(phrase_key, set()))
                   weighted_score = fact_score / max(ent_chunk_count, 1)
                   
                   phrase_weights[phrase_idx] += weighted_score
   ```

3. **Get DPR Scores cho Passages:**
   ```python
   dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
   normalized_dpr_scores = min_max_normalize(dpr_sorted_doc_scores)
   
   # Set passage weights
   for i, doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
       if i < len(self.knowledge_graph.passage_node_keys):
           passage_weights[doc_id] = normalized_dpr_scores[i] * self.config.passage_node_weight
   ```

4. **Combine Weights cho PPR:**
   ```python
   node_weights = self.knowledge_graph.get_node_weights_for_ppr(phrase_weights, passage_weights)
   ```

   **Chi tiết trong `graph.py`:**
   ```python
   def get_node_weights_for_ppr(self, phrase_weights, passage_weights):
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
   ```

5. **Run Personalized PageRank:**
   ```python
   ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.knowledge_graph.run_personalized_pagerank(
       node_weights, damping=self.config.damping
   )
   ```

   **Chi tiết PPR trong `graph.py`:**
   ```python
   def run_personalized_pagerank(self, reset_prob, damping=0.5):
       # Clean reset probabilities
       reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
       
       # Run PageRank với personalized reset
       pagerank_scores = self.graph.personalized_pagerank(
           vertices=range(len(self.node_name_to_vertex_idx)),
           damping=damping,
           directed=False,
           weights='weight',
           reset=reset_prob,
           implementation='prpack'
       )
       
       # Extract scores cho document nodes only
       doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
       sorted_doc_ids = np.argsort(doc_scores)[::-1]
       sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]
       
       return sorted_doc_ids, sorted_doc_scores
   ```

#### 3.2.4 Bước 4: Extract Top Documents

```python
top_docs = []
for doc_id in sorted_doc_ids[:num_to_retrieve]:
    if doc_id < len(self.knowledge_graph.passage_node_keys):
        passage_key = self.knowledge_graph.passage_node_keys[doc_id]
        doc_content = self.chunk_store.get_row(passage_key).get("content", "")
        top_docs.append(doc_content)

results.append(QuerySolution(
    question=query,
    docs=top_docs,
    doc_scores=sorted_doc_scores[:num_to_retrieve].tolist()
))
```

---

## 🤖 PHASE 4: QUESTION ANSWERING

### 4.1 QA Pipeline

```python
qa_solutions, response_messages, metadata = retriever.qa(query_solutions)
```

**Chi tiết trong `retrieval.py`:**

#### 4.1.1 Prepare QA Prompts

```python
all_qa_messages = []

for query_solution in query_solutions:
    retrieved_passages = query_solution.docs[:self.config.qa_top_k]
    
    prompt_user = ''
    for passage in retrieved_passages:
        prompt_user += f'Document: {passage}\n\n'
    prompt_user += f'Question: {query_solution.question}\nAnswer: '
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documents."},
        {"role": "user", "content": prompt_user}
    ]
    all_qa_messages.append(messages)
```

**Chi tiết Prompt Structure:**
```
System: You are a helpful assistant that answers questions based on the provided documents.

User: Document: Artificial intelligence is transforming the way we work and live. Machine learning algorithms can analyze vast amounts of data.

Document: Natural language processing enables computers to understand human language. Deep learning models have achieved remarkable results.

Document: Computer vision allows machines to interpret visual information. Neural networks are inspired by the human brain structure.

Question: What is artificial intelligence?
Answer: 
```

#### 4.1.2 Run LLM Inference

```python
all_qa_results = []
for qa_messages in all_qa_messages:
    try:
        result = self.llm_model.infer(qa_messages)
        all_qa_results.append(result)
    except Exception as e:
        logger.error(f"Error in QA inference: {e}")
        all_qa_results.append(("Error occurred", {}, False))

all_response_messages, all_metadata, all_cache_hits = zip(*all_qa_results)
```

#### 4.1.3 Extract Answers

```python
updated_solutions = []
for i, query_solution in enumerate(query_solutions):
    response = all_response_messages[i]
    
    # Extract answer từ response
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
```

### 4.2 Output Format

**Final QuerySolution:**
```python
QuerySolution(
    question="What is artificial intelligence?",
    docs=[
        "Artificial intelligence is transforming the way we work and live...",
        "Natural language processing enables computers to understand...",
        "Computer vision allows machines to interpret visual information..."
    ],
    doc_scores=[0.95, 0.87, 0.82],
    answer="Based on the provided documents, artificial intelligence is a technology that transforms the way we work and live by enabling machines to understand language, analyze data, and interpret visual information through various techniques like machine learning, natural language processing, and computer vision.",
    gold_answers=None,  # Nếu có ground truth
    gold_docs=None      # Nếu có ground truth documents
)
```

---

## 📊 PERFORMANCE TRACKING

### Timing Metrics

```python
# Được track trong retriever
retriever.total_retrieval_time  # Tổng thời gian retrieval
retriever.ppr_time             # Thời gian chạy PageRank
retriever.rerank_time          # Thời gian rerank facts

logger.info(f"Total Retrieval Time: {retriever.total_retrieval_time:.2f}s")
logger.info(f"Total Recognition Memory Time: {retriever.rerank_time:.2f}s") 
logger.info(f"Total PPR Time: {retriever.ppr_time:.2f}s")
```

### Graph Statistics

```python
graph_info = knowledge_graph.get_graph_info()

GraphInfo(
    num_phrase_nodes=1250,           # Số entity nodes
    num_passage_nodes=100,           # Số passage nodes  
    num_total_nodes=1350,            # Tổng nodes
    num_extracted_triples=890,       # Triples từ OpenIE
    num_triples_with_passage_node=100, # Edges passage-entity
    num_synonymy_triples=245,        # Synonymy edges
    num_total_triples=1235           # Tổng edges
)
```

---

## 🔧 ADVANCED FEATURES

### 1. Document Deletion

```python
success = indexer.delete_documents(documents_to_delete)
```

**Chi tiết process:**
1. Get chunk IDs để delete
2. Find entities và facts chỉ tồn tại trong deleted chunks
3. Remove từ all stores (chunk, entity, fact)
4. Remove vertices từ graph
5. Update OpenIE results
6. Save changes

### 2. Incremental Indexing

- Hệ thống tự động detect documents đã được index
- Chỉ process documents mới
- Merge results với existing data
- Efficient cho large-scale deployments

### 3. Fallback Mechanisms

- Nếu không có facts relevant → fallback về DPR
- Nếu OpenIE fails → tạo empty results
- Nếu graph empty → pure dense retrieval
- Error handling ở mọi level

### 4. Caching Strategy

- Query embeddings được cache
- Precomputed embeddings cho fast retrieval
- Metadata persistence để avoid recomputation
- File-based storage cho scalability

---

## 🎯 KEY ADVANTAGES

### 1. **Multi-hop Reasoning**
- Graph structure cho phép reasoning qua multiple hops
- Entity relationships được preserve
- Complex queries được handle tốt hơn

### 2. **Semantic Understanding**
- OpenIE extracts structured knowledge
- Entity embeddings capture semantic similarity
- Fact-based retrieval more precise than keyword matching

### 3. **Personalized Ranking**
- PPR personalizes ranking based on query
- Combines multiple signals (facts + dense retrieval)
- Adaptive weighting based on query characteristics

### 4. **Modular Design**
- Each component có thể optimize riêng
- Easy to swap different models
- Scalable architecture

---

## 🚀 OPTIMIZATION OPPORTUNITIES

### 1. **Batch Processing**
- Batch OpenIE calls cho efficiency
- Vectorized operations cho embeddings
- Parallel processing cho independent operations

### 2. **Memory Management**
- Lazy loading cho large embeddings
- Disk-based storage cho scalability
- Memory mapping cho fast access

### 3. **Caching**
- Query result caching
- Intermediate computation caching
- Smart cache invalidation

### 4. **Hardware Acceleration**
- GPU acceleration cho embeddings
- Optimized graph libraries
- Distributed processing cho large datasets

---

Đây là workflow siêu chi tiết của HippoRAG modular implementation. Mỗi bước được giải thích rõ ràng với code examples và data structures cụ thể. 