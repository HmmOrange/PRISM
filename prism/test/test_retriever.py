import asyncio

from prism.hg_store.retriever import Retriever

async def main():
    retriever = Retriever()
    documents = await retriever.retrieve("Use image-to-text to generate a description of the given image", task_type="image-to-text", top_k=10)
    print(len([{
        "score": document["score"],
        "hash_id": document["hash_id"],
    } for document in documents]))
    for idx, document in enumerate(documents):
        hash_id = document["hash_id"]
        print(f"{idx}. {hash_id}")
        print(document["content"])

if __name__ == '__main__': 
    asyncio.run(main())