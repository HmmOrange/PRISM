import re
import json
import asyncio
from mpmath import j
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, TypedDict

from utils.logs import logger
from utils.common import CodeParser

from prism.llm import LLM
from prism.schema.message import Message
from prism.prompts.prompt_template_manager import PromptTemplateManager
from prism.hg_store.typing import HuggingFaceItem

class InformationExtractor:
    def __init__(self):
        self.llm = LLM()
        self.prompt_template_manager = PromptTemplateManager()
        
    async def ner_query(self, query: str) -> dict:
        error_logs = []
        counter = 0
        max_retry = 3
        while counter < max_retry:
            ner_query_input_message = self.prompt_template_manager.render(name="ner_query", query=query)
            msg = self.llm.format_msg(ner_query_input_message + error_logs)
            response = await self.llm.aask(msg)
            extracted_entities = CodeParser.parse_code(response, lang="json")
            try:
                error_logs.append(Message(content=response, role="assistant"))
                extracted_entities = json.loads(extracted_entities)
                break
            except Exception as e:
                counter += 1
                error_logs.append(Message(content=str(e), role="user"))
        if not isinstance(extracted_entities, dict):
            raise Exception("Failed to extract entities")
        return extracted_entities
    
    async def description_hypothesis(self, query: str, extracted_entities: dict) -> str:
        description_hypothesis_input_messages = self.prompt_template_manager.render(
            name="model_description_hypothesis",
            query=query,
            named_entity_json=extracted_entities
        )
        msg = self.llm.format_msg(description_hypothesis_input_messages)
        response = await self.llm.aask(msg)
        return response
        
    async def ner(self, item: HuggingFaceItem) -> dict:
        print("Extracting information from item", item.id)
        ner_input_message = self.prompt_template_manager.render(name="ner", passage=str(item.to_dict())[:126000])
        msg = self.llm.format_msg(ner_input_message)
        response = await self.llm.aask(msg)
        extracted_entities = CodeParser.parse_code(response, lang="json")
        extracted_entities = json.loads(extracted_entities)
        return extracted_entities
        
    async def summarize_model_description(self, item: HuggingFaceItem) -> str:
        print("Summarizing model description for item", item.id)
        summarize_input_messages = self.prompt_template_manager.render(
            name="summarize_model_description",
            passage=str(item.to_dict())[:126000],
            named_entity_json=item.entity_extracted
        )
        msg = self.llm.format_msg(summarize_input_messages)
        response = await self.llm.aask(msg)
        return response


    async def information_extraction(self, item: HuggingFaceItem) -> HuggingFaceItem:
        
        ner_output = await self.ner(item)
        item.entity_extracted = ner_output
        summarize_output = await self.summarize_model_description(item)
        item.summary = summarize_output
        return item
    
    async def batch_information_extraction(self, items: List[HuggingFaceItem]) -> List[HuggingFaceItem]:
        """
        Batch information extraction with error handling per item
        If one item fails, others will continue processing
        """
        async def safe_information_extraction(item: HuggingFaceItem) -> HuggingFaceItem:
            """Wrapper function to handle individual item errors"""
            try:
                return await self.information_extraction(item)
            except Exception as e:
                logger.error(f"Failed to extract information for item {item.id}: {e}")
                # Return the original item without extracted information
                return None

        information_extraction_tasks = [
            safe_information_extraction(item)
            for item in items
        ]
        
        # Use gather with return_exceptions=False but handle exceptions in wrapper
        information_extraction_results_list = await asyncio.gather(*information_extraction_tasks)
        
        # Filter out None results and log success rate
        successful_results = [result for result in information_extraction_results_list if result is not None]
        success_rate = len(successful_results) / len(items) * 100
        logger.info(f"Batch extraction completed: {len(successful_results)}/{len(items)} items processed successfully ({success_rate:.1f}%)")
        
        return successful_results
    