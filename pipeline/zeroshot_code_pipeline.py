import json
from typing import Optional, Tuple

from templates.pipeline_template import Pipeline
from storage.model_storage import model_storage
from utils.cost_manager import CostManager
from configs.models_config import ModelsConfig
from provider.llm_provider_registry import create_llm_instance
from utils.common import CodeParser
import tiktoken
import random
from utils.common import describe_dict

generator_prompt = """
## Task

The main task is to create workflow using the existing models to solve the task. The workflow can be formulated into the equivalent Python code, where each statement represents the execution of a single node. You are an expert in the field of AI and model inference, helping workflows according to their requirements.

Now you are required to create a workflow for the following task:
<task>
{task} 
</task>

To use the existing models, you use model_inference function. Don't need to import model_inference function, just generate the code to use the model_inference function. If you use other functions, you need to import them.:

## Model informations
<model_info>
{model_info}
</model_info>

## Input/Output format
<input_output_format>
{input_output_format}
</input_output_format>

# Data Input Format
<data_input_format>
{data_input_format}
</data_input_format>

## To use model use function model_inference
response = model_inference(model_id=model_id, input_data=input_data, hosted_on=hosted_on, task=task)
model_inference is a sync function not used with await keyword
input_data is json object: {{"image": "...", "audio": "...", ...}}
response is json object: {{"predicted": "...", "text": "...", ...}}
hosted_on is "local" or "huggingface"
task is ["image-classification", "object-detection", "summarization", "translation", "sentence-similarity", "text-classification", "zero-shot-classification", "video-classification", "audio-classification", "automatic-speech-recognition", "image-to-text", "token-classification", "text-generation", "question-answering", "tabular-classification", "tabular-regression"]

## Format

You should provide your Python code to formulate the workflow. Each line of code should coresspond to a single node, so you should avoid nested calls in a single statement. 
You have to write code follow template:
<template>
from scripts.model_inference import model_inference
from typing import Union

class Workflow:
    async def __call__(self, data: dict):
        '''
        Process a single testcase and return prediction
        
        Args:
            data: {{
                "image_paths": [list of image file paths],
                "video_paths": [list of video file paths], 
                "audio_paths": [list of audio file paths],
                "text_data": {{dict of text data}}
            }}
            
        Returns:
            prediction: prediction for this testcase
        '''
</template>
IMPORTANT: 
- With text data, if the question does not say anything, it is in data["text_data"]["text"] by default, but if it is referred to as a record containing question, context, etc., it is in that field like data["text_data"]["question"], data["text_data"]["context"], etc.
- model_inference is a sync function not used with await keyword.
NOTE: Choose appropriate data type based on task - use only data["image_paths"] for image tasks, data["video_paths"] for video tasks, data["audio_paths"] for audio tasks, or multiple fields for multimodal tasks.

Your code should be following the format:
```python
# complete working code
from scripts.model_inference import model_inference
from typing import Union

class Workflow:
    async def __call__(self, data: dict):
        pass
```

Now, provide your code with the required format.
"""

llm = create_llm_instance(ModelsConfig.default().get("gpt-4o-mini"))
llm.cost_manager = CostManager()
tokenizer = tiktoken.get_encoding("cl100k_base")
MAX_TOKEN_CONTEXT = 120000
class ZeroShotCodePipeline(Pipeline):
    async def __call__(self, task_description: str, data: dict) -> Tuple[str, float]:
        model_documents = model_storage.get_all_models()
        documents = model_documents["documents"]
        random.shuffle(documents)
        model_info = ""
        validation_data = data["validation_data"]
        for i, document in enumerate(documents):
            model_info_tokens = tokenizer.encode(model_info, allowed_special="all")
            document_tokens = tokenizer.encode(document, allowed_special="all")
            if len(model_info_tokens) + len(document_tokens) > MAX_TOKEN_CONTEXT:
                break
            model_info += f"""<Model_{i}>
            {document}
            </Model_{i}>
            """
        model_usage_info = ""
        with open("data/model.json", "r", encoding="utf-8") as f:
            model_usage = json.load(f)
        for id, (model_type_name, model_type) in enumerate(model_usage.items()):
            model_usage_info += f"""{id}. Model type {model_type_name} With interface: {model_type["interface"]}. Output Format: {model_type["output"]}\n"""
        sample_data = validation_data.iloc[0].to_dict()
        del sample_data["input_id"]
        del sample_data["label"]
        prompt = generator_prompt.format(
            task=task_description,
            model_info=model_info,
            input_output_format=model_usage_info,
            data_input_format=describe_dict(sample_data)
        )
        response = await llm.aask(prompt)
        code = CodeParser.parse_code(text=response)
        return code, llm.cost_manager.get_total_cost() if llm.cost_manager else 0.0