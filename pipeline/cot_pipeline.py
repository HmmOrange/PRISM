from typing import Optional, Tuple
from templates.pipeline_template import Pipeline


workflow = """
import json
import base64
from utils.cost_manager import CostManager
from utils.logs import logger
from configs.models_config import ModelsConfig
from openai import OpenAI
from templates.cot_prompt import COT_PROMPT, parse_prediction
from utils.constants import COMPLETION_API_KEY
from utils.token_counter import TOKEN_COSTS
from provider.llm_provider_registry import create_llm_instance

client = OpenAI(api_key=COMPLETION_API_KEY)

llm = create_llm_instance(ModelsConfig.default().get("gpt-4o-mini"))
llm.cost_manager = CostManager()

def update_total_cost(task_name: str, usage):
    try:
        with open("data/cot_cost.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        data = {{}}
    total_cost = data.get(task_name, 0)
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    current_cost = (prompt_tokens * TOKEN_COSTS["gpt-4o-mini"]["prompt"] + completion_tokens * TOKEN_COSTS["gpt-4o-mini"]["completion"]) / 1000
    total_cost += current_cost
    logger.info(f"Total prompt tokens: {{prompt_tokens}} | Total completion tokens: {{completion_tokens}} |  Running cost: ${{current_cost:.3f}} | Total running cost: ${{total_cost:.3f}}")
    data[task_name] = total_cost 
    with open("data/cot_cost.json", "w", encoding="utf-8") as f:
        json.dump(data, f)

class Workflow:
    async def __call__(self, data: dict) -> str:
        image_paths = data.get("image_paths", [])
        video_paths = data.get("video_paths", [])
        audio_paths = data.get("audio_paths", [])
        text_data = data.get("text_data", {{}})
        task_description = '''{task_description}'''
        task_name = '''{task_name}'''
        if len(video_paths) > 0 or len(audio_paths) > 0:
            return ""
        messages =[{{"role": "system",
                    "content": "You are a helpful assistant."}}]
        usr_msg = {{
            "role": "user",
            "content": [{{
                "type": "text",
                "text": COT_PROMPT.format(description=task_description, input_text=text_data)
            }}]
        }}
        for image_path in image_paths:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
                usr_msg["content"].append({{
                    "type": "image_url",
                    "image_url": {{
                        "url": f"data:image/png;base64,{{image_base64}}"   
                    }}
                }})
        messages.append(usr_msg)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        content = response.choices[0].message.content
        logger.info(f"Prediction: {{content}}")
        usage = response.usage
        update_total_cost(task_name, usage)
        prediction = parse_prediction(content)
        return prediction
"""

class CoTPipeline(Pipeline):
    async def __call__(self, task_description: str, data: dict) -> Tuple[str, float]:
        task_name = data["task_name"]
        return workflow.format(task_description=task_description, task_name=task_name), 0



