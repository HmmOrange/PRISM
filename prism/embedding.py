from typing import Optional

from configs.embedding_config import EmbeddingConfig
from prism.schema.context import Context
from provider.embedding.base_embedding import BaseEmbedding

def EMBEDDING(embedding_config: Optional[EmbeddingConfig] = None, context: Optional[Context] = None) -> BaseEmbedding:
    ctx = context or Context()
    if embedding_config is None:
        return ctx.embedding_with_cost_manager_from_embedding_config(ctx.config.embedding)
    return ctx.embedding()
