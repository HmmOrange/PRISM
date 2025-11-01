import asyncio
from pydantic import Field, SerializeAsAny, BaseModel, ConfigDict
from typing import Iterable, Dict, Set, Any, Optional, Union

from utils.logs import logger
from utils.common import is_send_to

from agent.base.base_env import BaseEnvironment
from agent.base.base_env_space import BaseEnvObsParams, BaseEnvAction
from agent.schema.message import Message
from agent.base.base_agent import BaseAgent
from agent.schema.context import Context
from agent.schema.memory import Memory
from agent.schema.plan import Plan
from agent.environment.nb_workspace import NbCodeWorkspace

class CAEnv(BaseEnvironment, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    nb_workspace: NbCodeWorkspace = Field(default_factory=NbCodeWorkspace)
    desc: str = Field(default="")
    agents: dict[str, SerializeAsAny[BaseAgent]] = Field(default_factory=dict, validate_default=True)
    member_addrs: Dict[BaseAgent, Set] = Field(default_factory=dict, exclude=True)
    history: Memory = Field(default_factory=Memory)
    context: Context = Field(default_factory=Context)
    plan: Optional[Plan] = Field(default=None)
    task_done: list[str] = Field(default_factory=list)

    final_code: str = Field(default="")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {}

    def observe(self, obs_params: Optional[BaseEnvObsParams] = None) -> Any:
        return None

    def step(self, action: BaseEnvAction) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return {}, 0.0, False, False, {}
   
    def set_task_info(self, input_desc: str, problem_desc: str,  output_desc: str):
        self.input_desc = input_desc
        self.problem_desc = problem_desc
        self.output_desc = output_desc
    
    def add_agent(self, agent: BaseAgent):
        self.agents[agent.name] = agent
        agent.set_env(self)
        agent.context = self.context
    
    def add_agents(self, agents: Iterable[BaseAgent]):
        for agent in agents:
            self.agents[agent.name] = agent
        for agent in agents:
            agent.context = self.context
            agent.set_env(self)
    
    def publish_message(self, message: Message) -> bool:
        """
        Distribute the message to the recipients.
        In accordance with the Message routing structure design in Chapter 2.2.1 of RFC 116, as already planned
        in RFC 113 for the entire system, the routing information in the Message is only responsible for
        specifying the message recipient, without concern for where the message recipient is located. How to
        route the message to the message recipient is a problem addressed by the transport framework designed
        in RFC 113.
        """
        logger.debug(f"publish_message: {message.dump()}")
        found = False
        # According to the routing feature plan in Chapter 2.2.3.2 of RFC 113
        for role, addrs in self.member_addrs.items():
           
            if is_send_to(message, addrs):
                role.put_message(message)
                found = True
        if not found:
            logger.warning(f"Message no recipients: {message.dump()}")
        self.history.add(message)  # For debug

        return True    
    
    def get_agents(self) -> dict[str, BaseAgent]:
        return self.agents
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        return self.agents.get(name, None)
    
    def agent_names(self) -> list[str]:
        return [agent.name for agent in self.agents.values()]
    
    @property
    def is_idle(self):
        """If true, all actions have been executed."""
        for agent in self.agents.values():
            if not agent.is_idle:
                return False
        return True
    
    def get_addresses(self, obj):
        """Get the addresses of the object."""
        return self.member_addrs.get(obj, {})

    def set_addresses(self, obj, addresses):
        """Set the addresses of the object"""
        self.member_addrs[obj] = addresses
    
    async def run(self, k = 1):
        """Process all Agent runs at once"""
        for _ in range(k):
            futures = []
            agents = []
            for agent in self.agents.values():
                if agent.is_idle:
                    # print(f"agent {agent.name} is idle")
                    continue
                else:
                    print(f"agent {agent.name} is observing")
                agents.append(agent)
                
                future = agent.run()
                futures.append(future)
            
            if futures:
                await asyncio.gather(*futures)
            logger.debug(f"is idle: {self.is_idle}")
            