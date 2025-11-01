# HippoRAG Modular Implementation

Đây là phiên bản modular của HippoRAG, được chia nhỏ thành các modules có tính logic để dễ quản lý và phát triển.

## Cấu trúc thư mục

```
huggingface_storage/
├── __init__.py           # Export các classes chính
├── typing.py             # Định nghĩa data types và configs
├── stores.py             # Quản lý embedding storage và OpenIE results
├── utils.py              # Utility functions
├── graph.py              # Knowledge graph operations
├── indexing.py           # Document indexing operations
├── retrieval.py          # Retrieval và QA operations
├── demo.py               # Demo script
└── README.md             # Tài liệu này
```

## Các modules chính

### 1. `typing.py` - Data Types & Configs
- `QuerySolution`: Kết quả query với documents và scores
- `NerRawOutput`: Kết quả Named Entity Recognition
- `TripleRawOutput`: Kết quả triple extraction
- `RetrievalConfig`: Configuration cho retrieval
- `GraphInfo`: Thông tin thống kê về graph
- `IndexingResult`: Kết quả quá trình indexing

### 2. `stores.py` - Storage Management
- `EmbeddingStore`: Quản lý embeddings cho chunks, entities, facts
- `OpenIEResultsManager`: Quản lý kết quả OpenIE extraction
- `reformat_openie_results()`: Helper function để format OpenIE results

### 3. `utils.py` - Utility Functions
- `compute_mdhash_id()`: Tính MD5 hash với prefix
- `text_processing()`: Xử lý text cơ bản
- `extract_entity_nodes()`: Trích xuất entities từ triples
- `flatten_facts()`: Flatten list triples
- `min_max_normalize()`: Normalize scores
- `retrieve_knn()`: KNN search cho similarity

### 4. `graph.py` - Knowledge Graph
- `KnowledgeGraph`: Quản lý knowledge graph chính
  - Tạo và load graph từ file
  - Thêm fact edges, passage edges, synonymy edges
  - Chạy Personalized PageRank
  - Quản lý nodes và mappings

### 5. `indexing.py` - Document Indexing
- `HippoIndexer`: Engine cho việc index documents
  - Index documents thành embeddings
  - Thực hiện OpenIE extraction
  - Xây dựng knowledge graph
  - Hỗ trợ xóa documents

### 6. `retrieval.py` - Retrieval & QA
- `HippoRetriever`: Engine cho retrieval và QA
  - Fact-based retrieval với graph search
  - Dense passage retrieval
  - Reranking facts
  - Question answering

## Cách sử dụng

### Cơ bản

```python
from huggingface_storage import RetrievalConfig
from huggingface_storage.indexing import HippoIndexer
from huggingface_storage.retrieval import HippoRetriever

# 1. Setup configuration
config = RetrievalConfig(
    retrieval_top_k=10,
    linking_top_k=5,
    qa_top_k=3
)

# 2. Initialize indexer
indexer = HippoIndexer(
    working_dir="./data",
    openie_model=your_openie_model,
    embedding_model=your_embedding_model,
    config=config
)

# 3. Index documents
documents = ["Document 1", "Document 2", "Document 3"]
result = indexer.index_documents(documents)
print(f"Indexed {result.num_docs_indexed} documents")

# 4. Initialize retriever
embedding_stores = indexer.get_embedding_stores()
knowledge_graph = indexer.get_knowledge_graph()

retriever = HippoRetriever(
    knowledge_graph=knowledge_graph,
    embedding_stores=embedding_stores,
    llm_model=your_llm_model,
    config=config
)

# 5. Perform retrieval
queries = ["Query 1", "Query 2"]
results = retriever.retrieve(queries, your_embedding_model)

# 6. Perform QA
qa_results, responses, metadata = retriever.qa(results)
for solution in qa_results:
    print(f"Q: {solution.question}")
    print(f"A: {solution.answer}")
```

### Chạy demo

```python
# Chạy demo với mock models
from huggingface_storage.demo import demo_hippo_rag
demo_hippo_rag()
```

## Ưu điểm của thiết kế modular

1. **Tách biệt concerns**: Mỗi module có trách nhiệm rõ ràng
2. **Dễ test**: Có thể test từng module riêng biệt
3. **Dễ maintain**: Thay đổi một module không ảnh hưởng modules khác
4. **Tái sử dụng**: Các modules có thể được sử dụng độc lập
5. **Mở rộng**: Dễ dàng thêm tính năng mới hoặc thay thế implementation

## Tích hợp với code gốc

Các modules này được thiết kế để tương thích với code HippoRAG gốc:

- Interfaces giống nhau
- Data structures tương thích
- Performance tương đương
- Hỗ trợ tất cả tính năng chính

## Dependencies

- `numpy`: Cho mathematical operations
- `igraph`: Cho graph operations
- `sklearn`: Cho KNN search
- `tqdm`: Cho progress bars
- `json`, `os`, `logging`: Standard library

## Lưu ý

- Code này là implementation mẫu, có thể cần điều chỉnh cho use case cụ thể
- Mock models trong demo chỉ để minh họa, cần thay thế bằng models thật
- Một số lỗi typing nhỏ có thể cần fix tùy theo Python version
- Cần có igraph package: `pip install python-igraph` 