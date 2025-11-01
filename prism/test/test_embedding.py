import asyncio

from prism.settings import prism_embedding_model
from prism.settings import cost_manager

async def test_embedding():
    embedding = prism_embedding_model
    texts=["Xin chào"]
    print(len(texts))
    print(len(await embedding.batch_encode(texts)))
    # print(cost_manager.get_total_cost())


if __name__ == "__main__":
    asyncio.run(test_embedding())