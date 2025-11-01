import os
import json
import re
import pandas as pd
from bs4 import BeautifulSoup
from typing import Callable, List, Tuple
from pydantic import Field, model_validator, BaseModel, ConfigDict
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing_extensions import Self
from time import time

from scripts.calculator import Calculator, MetricType
from prism.settings import cost_manager
from utils.common import general_after_log
from utils.logs import logger
from templates.pipeline_template import Pipeline
from provider.llm.base_llm import BaseLLM

from prism.llm import LLM
from prism.prompts.prompt_template_manager import PromptTemplateManager
from prism.ext.mas.utils.convergence_utils import ConvergenceUtils
from prism.ext.mas.utils.workflow_utils import WorkflowUtils
from prism.ext.mas.utils.experience_utils import ExperienceUtils
from prism.ext.mas.utils.data_utils import DataUtils
from prism.ext.mas.utils.evaluation_utils import EvaluationUtils

DEFAULT_PATH = "prism/ext/mas"


class masPipeline(Pipeline, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: BaseLLM = Field(default_factory=LLM)
    model_type_description: str = ""
    agent_description: str = ""
    prompt_template_manager: PromptTemplateManager = Field(
        default_factory=lambda: PromptTemplateManager(
            template_dirs=["prism/ext/mas/prompts"]
        )
    )
    default_path: str = "prism/ext/mas"
    task_name: str = ""
    level: str = ""
    calculator: Calculator = Field(default_factory=Calculator)
    workflow_utils: WorkflowUtils = Field(
        default_factory=lambda: WorkflowUtils(root_path=DEFAULT_PATH)
    )
    experience_utils: ExperienceUtils = Field(
        default_factory=lambda: ExperienceUtils(root_path=DEFAULT_PATH)
    )
    data_utils: DataUtils = Field(
        default_factory=lambda: DataUtils(root_path=DEFAULT_PATH)
    )
    evaluation_utils: EvaluationUtils = Field(
        default_factory=lambda: EvaluationUtils(root_path=DEFAULT_PATH)
    )
    convergence_utils: ConvergenceUtils = Field(
        default_factory=lambda: ConvergenceUtils(root_path=DEFAULT_PATH)
    )

    async def __call__(self, task_description: str, data: dict) -> Tuple[str, float]:
        cost_manager.reset()


        self.level = data["level"]
        self.task_name = data["task_name"]
        workflow_path = os.path.join(
            self.default_path, "workflows_direct_mas", self.level, self.task_name
        )
        results_path = self.data_utils.get_results_file_path(workflow_path, reset=False)
        template_path = os.path.join(self.default_path, "templates")

        model_type_description = self.data_utils.load_model_type_description(
            template_path, "<all>"
        )
        print(model_type_description)
        results = self.data_utils.get_results(workflow_path)
        
        default_iteration = 1
        
        cost_manager.reset()
        start_time = time()
        validation_data = data["validation_data"]
        validation_data_5 = (
            validation_data.sample(5)
            if validation_data.shape[0] > 5
            else validation_data
        )


        graph = await self.workflow_utils.generate_direct_mas(
            task_description=task_description,
            data=validation_data_5,
            model_type_description=model_type_description,
        )

        self.workflow_utils.write_workflow_file(
            graph, default_iteration, workflow_path, is_solution=False
        )

        graph = self.workflow_utils.read_workflow_file(
            default_iteration, workflow_path, is_solution=False
        )

        workflow_instance = self.workflow_utils.load_workflow(default_iteration, workflow_path)
        workflow_instance = workflow_instance(
            problem=task_description, data=validation_data_5
        )
        code = await workflow_instance()
        self.workflow_utils.write_workflow_file(
            code, default_iteration, workflow_path, is_solution=True
        )

        code = self.workflow_utils.read_workflow_file(
            default_iteration, workflow_path, is_solution=True
        )
        code_solution = self.workflow_utils.load_workflow(
            default_iteration, workflow_path, is_solution=True
        )

        avg_score, logs = await self.evaluation_utils.evaluate_workflow(
            workflow_class=code_solution,
            directory=workflow_path,
            data=validation_data_5,
            label=data["label"],
            metric=data["metric"],
        )

        end_time = time()

        result = self.data_utils.create_result_data(
            default_iteration,
            avg_score,
            cost_manager.get_total_cost(),
            generation_time=(end_time - start_time),
            code=code,
        )
        results.append(result)
        self.data_utils.save_results(results_path, results)

        experience = self.experience_utils.create_experience_data(
            avg_score=avg_score,
            sketch="",
            workflow=graph,
            logs=logs,
            round=default_iteration,
            review="",
            cost=cost_manager.get_total_cost(),
            generation_time=end_time - start_time,
            generated_code=code,
        )

        self.experience_utils.update_experience(workflow_path, default_iteration, experience)

        best_round = await self.data_utils.select_best_round(
            workflow_path, default_iteration, user_requirement=task_description
        )
        best_solution = self.workflow_utils.read_workflow_file(
            best_round, workflow_path, is_solution=True
        )

        return best_solution, sum([result["cost"] for result in results])
