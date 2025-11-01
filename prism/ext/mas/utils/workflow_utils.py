import os
import re
import json
from typing import Callable, Tuple
import pandas as pd
from bs4 import BeautifulSoup
from tenacity import retry, wait_random_exponential, stop_after_attempt

from utils.logs import logger
from utils.common import describe_dict, general_after_log, CodeParser

from prism.prompts.prompt_template_manager import PromptTemplateManager
from prism.llm import LLM
from prism.schema.message import Message

class WorkflowUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.prompt_template_manager = PromptTemplateManager(template_dirs=[os.path.join(root_path, "prompts")])
        self.llm = LLM()
            
    def create_round_directory(self, workflow_path: str, round_number: int) -> str:
        directory = os.path.join(workflow_path, f"round_{round_number}")
        os.makedirs(directory, exist_ok=True)
        return directory
    
    def read_mermaid_file(self, round_number: int, workflow_path: str) -> str:
        mermaid_file_path = os.path.join(workflow_path, f"round_{round_number}", "sketch.md")
        with open(mermaid_file_path, "r", encoding="utf-8") as file:
            mermaid_content = file.read()
        return mermaid_content
    
    def read_workflow_file(self, round_number: int, workflow_path: str, is_solution: bool = False) -> str:
        workflow_file_path = os.path.join(workflow_path, f"round_{round_number}", f"workflow{'_solution' if is_solution else ''}.py")
        with open(workflow_file_path, "r", encoding="utf-8") as file:
            workflow_content = file.read()
        return workflow_content
        
    def extract_solve_workflow(self, workflow_content: str) -> str:
        pattern = r"class Workflow:.+"
        return re.findall(pattern, workflow_content, re.DOTALL)[0]
    
    def load_workflow(self, round_number: int, workflow_path: str, is_solution: bool = False) -> Callable:
        directory = os.path.join(workflow_path, f"round_{round_number}")
        directory = directory.replace("\\", ".").replace("/", ".")
        
        workflow_module_name = f"{directory}.workflow{'_solution' if is_solution else ''}"
        try:
            workflow_module = __import__(workflow_module_name, fromlist=[""])
            workflow_class = getattr(workflow_module, "Workflow")
            return workflow_class
        except ImportError as e:
            logger.error(f"Error loading workflow for round {round_number}: {e}")
            raise
        
    def write_workflow_file(self, workflow_content: str, round_number: int, workflow_path: str, is_solution: bool = False):
        directory = os.path.join(workflow_path,  f"round_{round_number}")
        os.makedirs(directory, exist_ok=True)
        
        workflow_file_path = os.path.join(directory, f"workflow{'_solution' if is_solution else ''}.py")
        
        with open(workflow_file_path, "w", encoding="utf-8") as file:
            file.write(workflow_content)
        
    def write_mermaid_file(self, mermaid_content: str, round_number: int, workflow_path: str):
        mermaid_file_path = os.path.join(workflow_path, f"round_{round_number}", "sketch.md")
        with open(mermaid_file_path, "w", encoding="utf-8") as file:
            file.write(mermaid_content)
        
    def parse_xml(self, text: str, args: list[str], json_args: list[str] = []) -> dict:
        soup = BeautifulSoup(text, "html.parser")
        res = {}
        for arg in args:
            field = soup.find(arg)
            if not field:
                raise Exception(f"Field {arg} not found in the xml")
            res[arg] = field.get_text(strip=True)
        for json_arg in json_args:
            res[json_arg] = json.loads(res[json_arg])
            
        return res
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3), after=general_after_log(logger))
    async def generate_mas(self, task_description: str, data: pd.DataFrame, experience: str, round_number: int, model_type_description: str) -> Tuple[str, str]:
        sample = data.iloc[0].to_dict()
        generation_memory = [
            
        ]
        if round_number == 1:
            sketch_generation_input = self.prompt_template_manager.render(name="sketch_generation", model_type_description=model_type_description, user_requirement=task_description, data=describe_dict(sample), experience=experience)
        else:
            sketch_generation_input = self.prompt_template_manager.render(name="sketch_evolution", model_type_description=model_type_description, user_requirement=task_description, data=describe_dict(sample), experience=experience)
        sketch_generation_msg = self.llm.format_msg(sketch_generation_input)
        sketch_generation_rsp = await self.llm.aask(sketch_generation_msg)
        # sketch_generation_rsp = self.parse_xml(sketch_generation_rsp, ["sketch"])
        sketch = CodeParser.parse_code(sketch_generation_rsp)
        mas_generation_input = self.prompt_template_manager.render(name="workflow_generation", user_requirement=task_description,
                                sketch=sketch)
        mas_generation_msg = self.llm.format_msg(mas_generation_input)
        mas_generation_rsp = await self.llm.aask(mas_generation_msg)
        workflow = CodeParser.parse_code(mas_generation_rsp)
        return sketch, workflow
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3), after=general_after_log(logger))
    async def generate_direct_mas(self, task_description: str, data: pd.DataFrame, model_type_description: str) -> str:
        sample = data.iloc[0].to_dict()
        mas_generation_input = self.prompt_template_manager.render(name="direct_workflow_generation", model_type_description=model_type_description, user_requirement=task_description, data=describe_dict(sample))
        mas_generation_msg = self.llm.format_msg(mas_generation_input)
        mas_generation_rsp = await self.llm.aask(mas_generation_msg)
        mas_generation_rsp = self.parse_xml(mas_generation_rsp, ["graph"])
        workflow = mas_generation_rsp["graph"]
        return workflow
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3), after=general_after_log(logger))
    async def review_mas(self, task_description: str, model_type_description: str, sketch: str, code: str, score: float, logs: list[str]):
        review_mas_input = self.prompt_template_manager.render(name="llm_judge", user_requirement=task_description, model_type_description=model_type_description, sketch=sketch, code=code, score=score, logs=logs[:5])
        review_mas_msg = self.llm.format_msg(review_mas_input)
        review_mas_rsp = await self.llm.aask(review_mas_msg)
        review_mas_rsp = self.parse_xml(review_mas_rsp, ["review"])
        return review_mas_rsp["review"]