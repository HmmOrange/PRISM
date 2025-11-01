import asyncio

from configs.models_config import ModelsConfig
from provider.llm_provider_registry import create_llm_instance

from prism.llm import LLM

async def test_llm():
    llm = LLM()
    await llm.aask("Who are you?")

if __name__ == '__main__':
    asyncio.run(test_llm())