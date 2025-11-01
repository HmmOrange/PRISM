from typing import Optional, Any, List
from pathlib import Path
from pydantic import BaseModel, Field
import random
import pandas as pd
from itertools import islice

from templates.pipeline_template import Pipeline

from utils.common import NoMoneyException
from utils.logs import logger

from agent.actions.user_requirement import UserRequirement
from agent.schema.message import Message
from agent.schema.context import Context
from agent.environment.ca_env import CAEnv
from agent.base.base_agent import BaseAgent
from agent.planner import Planner
from agent.ml_coder import MLCoder
from agent.logic_coder import LogicCoder
from agent.synthesizer import Synthesizer
from agent.data_loader import DataLoader


class CAPipeline(Pipeline, BaseModel):
    max_round: int = 30
    env: Optional[CAEnv] = None
    investment: float = Field(default=3.0)
    n_round: int = 0
    planner: Planner = Field(default_factory=Planner)

    def __init__(self,  **data: Any):
        super(CAPipeline, self).__init__(**data)

    async def init_pipeline(self):
        self.n_round = self.max_round
        self.env = CAEnv(context=Context())
        self._setup_agents()
        
    def _setup_agents(self):
        """Create and add agents to environment"""
        self.planner = Planner()
        ml_coder = MLCoder()
        logic_coder = LogicCoder()
        synthesizer = Synthesizer()
        data_loader = DataLoader()
        
        self.add_agents([self.planner,
                         data_loader,
                         ml_coder, logic_coder,
                         synthesizer])
            
    @property
    def cost_manager(self):
        """Get cost manager"""
        if not self.env:
            raise ValueError("Environment not initialized")
        return self.env.context.cost_manager

    def invest(self, investment: float):
        """Invest budget. raise NoMoneyException when exceed max_budget."""
        self.investment = investment
        self.cost_manager.max_budget = investment
        logger.info(f"Investment: ${investment}.")

    def _check_balance(self):
        if self.cost_manager.total_cost >= self.cost_manager.max_budget:
            raise NoMoneyException(
                self.cost_manager.total_cost,
                f"Insufficient funds: {self.cost_manager.max_budget}",
            )

    def add_agents(self, agents: List[BaseAgent]):
        if not self.env:
            raise ValueError("Environment not initialized")
        self.env.add_agents(agents)

    async def __call__(
        self, task_description: str, data: Optional[dict] = None
    ) -> str:
        await self.init_pipeline()
        if not self.env:
            raise ValueError("Environment not initialized")
        if not data:
            raise ValueError("Metadata not provided")
        self.env.sample_data = data.get("validation_data", [])[0]
        items = data.get("validation_data", [])
        self.env.validation_data = random.sample(items, min(5, len(items)))

        # Merge label into validation_data if label is DataFrame
        if hasattr(self.env.label, 'iterrows'):
            for _, row in self.env.label.iterrows():
                data_id = str(row['id'])
                if data_id in self.env.validation_data:
                    # Convert numpy types to Python native types
                    label_value = row['label']
                    if hasattr(label_value, 'item'):  # numpy scalar
                        label_value = label_value.item()
                    self.env.validation_data[data_id]['label'] = label_value

        self.env.user_requirement = task_description
        self.env.publish_message(Message(content=task_description, cause_by=UserRequirement, sent_from="User", send_to="Data Loader"))
        while self.n_round > 0:
            if self.env.is_idle:
                logger.info("All agents are idle, stopping the pipeline")
                break
            self.n_round -= 1

            await self.env.run()

            logger.debug(f"max {self.n_round=} left.")
        if self.env.final_code == "":
            msg = Message(
                send_to={"Synthesizer"},
                metadata={
                    "plan_status": self.planner.get_plan_status()
                }
            )
            self.env.publish_message(msg)
            await self.env.run()
        return self.env.final_code

