import json
from copy import deepcopy
from bs4 import BeautifulSoup
import pandas as pd
from typing import Optional, Literal
from pydantic import Field, model_validator
from typing_extensions import Self

from prism.prompts import prompt_template_manager
from utils.logs import logger
from utils.common import CodeParser

from prism.utils.report import ThoughtReporter
from prism.schema.message import Message
from prism.prompts.prompt_template_manager import PromptTemplateManager
from prism.base.base_agent import BaseAgent
from prism.tools.tool_recommend import ToolRecommender, BM25ToolRecommender
from prism.actions.user_requirement import UserRequirement
from prism.actions.execute_nb_code import ExecuteNbCode 

from prism.ext.mas.actions.synthesize_code import SynthesizeCode

REACT_THINK_PROMPT = """
# User Requirement
{user_requirement}
# Context
{context}

Output a json following the format:
```json
{{
    "thoughts": str = "Thoughts on current situation, reflect on how you should proceed to fulfill the user requirement",
    "state": bool = "Decide whether you need to take more actions to complete the user requirement. Return true if you think so. Return false if you think the requirement has been completely fulfilled."
}}
```
"""

LOAD_DATA_CODE = """
import pandas as pd
data.head()
"""

PRINT_FIRST_ROW_CODE = """
if 'data' in locals() and not data.empty:
    print("First row of data:")
    first_row = data.iloc[0]
    print(first_row)
    print("\\nData types of first row:")
    for col, value in first_row.items():
        print(f"{col}: {value} (type: {type(value).__name__})")
else:
    print("No data available or data is empty")
"""

class Coder(BaseAgent):
    name: str = "Coder"
    profile: str = "A professional coder"
    auto_run: bool = True
    use_plan: bool = False
    use_reflection: bool = False
    execute_code: ExecuteNbCode = Field(default_factory=ExecuteNbCode, exclude=True)
    tools: list[str] = ["<all>"]
    react_mode: Literal["plan_and_act", "react"] = "react"
    max_react_loop: int = 5  # used for react mode
    user_requirement: str = ""
    model_description: str = ""
    tool_recommender: Optional[ToolRecommender] = None
    prompt_template_manager: PromptTemplateManager = Field(
        default_factory=lambda: PromptTemplateManager(
            template_dirs=["prism/ext/mas/prompts"]
        )
    )
    experience: list[Message] = []
    problem: str = ""
    data: pd.DataFrame = pd.DataFrame()
    react_think_prompt: str = REACT_THINK_PROMPT
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @model_validator(mode="after")
    def set_plan_and_tool(self) -> Self: 
        self._set_react_mode(react_mode=self.react_mode, max_react_loop=self.max_react_loop, auto_run=self.auto_run)
        self.use_plan = (
            self.react_mode == "plan_and_act"
        )
        if self.tools and not self.tool_recommender:
            self.tool_recommender = BM25ToolRecommender(tools=self.tools)
        return self
    
    @property
    def working_memory(self):
        return self.ac.working_memory
    
    @property
    def longterm_memory(self):
        return self.ac.memory
    
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
    
    async def _think(self) -> bool:
        """Useful in 'react' mode. Use LLM to decide whether and what to do next."""
        context = self.working_memory.get()

        if not context:
            # just started the run, we need action certainly
            self.working_memory.add(self.get_memories()[0])  # add user requirement to working memory
            self._set_state(0)
            return True
        counter = 0
        max_retry = 3
        think_logs = []
        while counter < max_retry:
            try:
                prompt = self.react_think_prompt.format(user_requirement=self.user_requirement, context=context)
                msg = self.llm.format_msg(prompt)
                async with ThoughtReporter(enable_llm_stream=True):
                    rsp = await self.llm.aask(msg + think_logs)
                think_logs.append(Message(content=rsp, role="assistant"))
                rsp_dict = json.loads(CodeParser.parse_code(text=rsp))
                break
            except Exception as e:
                think_logs.append(Message(content=str(e), role="user"))
                counter += 1
                continue
        self.working_memory.add(Message(content=rsp_dict["thoughts"], role="assistant"))
        need_action = rsp_dict["state"]
        self._set_state(0) if need_action else self._set_state(-1)

        return need_action

    async def _act(self) -> Message:
        """Useful in 'react' mode. Return a Message conforming to Role._act interface."""
        code, _, _ = await self._write_and_exec_code()
        return Message(content=code, role="assistant", sent_from=self._setting, cause_by=self.todo)
    
    async def _react(self) -> Message:
        """Think first, then act, until the Role _think it is time to stop and requires no more todo.
        This is the standard think-act loop in the ReAct paper, which alternates thinking and acting in task solving, i.e. _think -> _act -> _think -> _act -> ...
        Use llm to select actions in _think dynamically
        """
        actions_taken = 0
        while actions_taken < self.ac.max_react_loop:
            # think
            has_todo = await self._think()
            if not has_todo:
                break
            # act
            logger.debug(f"{self._setting}: {self.ac.state=}, will do {self.ac.todo}")
            await self._act()
            actions_taken += 1
        code, output, success = await self._synthesize_code()
        return Message(content=code, role="assistant", metadata={
            "output": output,
            "success": success,
            "instruction": self.user_requirement,
            "previous_experience": deepcopy(self.experience)
        })
        
    async def _synthesize_code(self, max_retry: int = 3):
        counter = 0
        success = False
        
        while not success and counter < max_retry:
            code = await SynthesizeCode().run(
                user_requirement=self.user_requirement,
                working_memory=self.working_memory.get(),
                longterm_memory=self.longterm_memory.get(),
                prompt_template_manager=self.prompt_template_manager
            )
            self.working_memory.add(Message(content=code, role="assistant", cause_by=SynthesizeCode))
            output, success = await self.execute_code.run(code)
            self.working_memory.add(Message(content=output, role="user", cause_by=ExecuteNbCode))
            counter += 1
            
        return code, output, success
    
    async def load_data(self):
        output, success = await self.execute_code.run(f"""import pandas as pd
data = {self.data.to_dict()}
data = pd.DataFrame(data)""")
        self.longterm_memory.add(Message(content=LOAD_DATA_CODE,  role="assistant"))
        output, success = await self.execute_code.run(LOAD_DATA_CODE)
        if not success:
            raise Exception("Failed to load data")
        self.longterm_memory.add(Message(content=output, role="user"))

    async def print_first_row(self):
        self.longterm_memory.add(Message(content=PRINT_FIRST_ROW_CODE, role="assistant"))
        output, success = await self.execute_code.run(PRINT_FIRST_ROW_CODE)
        if not success:
            raise Exception("Failed to print first row")
        self.longterm_memory.add(Message(content=output, role="user"))
    
    async def load_experience(self):
        """Load experience recursively from oldest to newest, avoiding duplicates"""
        
        def collect_all_experiences(experiences: list[Message]) -> list[Message]:
            """Recursively collect all experiences, starting from the oldest"""
            all_old_experiences = []
            current_experiences = []
            
            for exp in experiences:
                # First, collect all previous experiences (older ones) recursively
                previous_experiences = exp.metadata.get('previous_experience')
                if previous_experiences:
                    # Recursively get older experiences first
                    older_experiences = collect_all_experiences(previous_experiences)
                    all_old_experiences.extend(older_experiences)
                
                # Add current experience to the current list
                current_experiences.append(exp)
            
            # Return all old experiences first, then current experiences
            return all_old_experiences + current_experiences
        
        # Collect all experiences from oldest to newest
        all_experiences = collect_all_experiences(self.experience)
        
        # Track loaded instructions to avoid duplicates
        loaded_instructions = set()
        
        # Load experiences in chronological order (oldest first)
        for msg in all_experiences:
            code = msg.content
            metadata = msg.metadata
            instruction = metadata["instruction"]
            
            # Skip if this instruction has already been loaded
            if instruction in loaded_instructions:
                continue
                
            # Add to tracking set
            loaded_instructions.add(instruction)
            
            # Load the experience
            self.longterm_memory.add(Message(content=instruction, role="user"))
            self.longterm_memory.add(Message(content=code, role="assistant"))
            output, success = await self.execute_code.run(code)
            self.longterm_memory.add(Message(content=output, role="user"))
            
    async def run(self, instruction: str, experience: list[Message] = []) -> Message:
        self.experience = experience
        await self.execute_code.init_code()
        await self.load_data()
        await self.print_first_row()
        await self.load_experience()
        self.user_requirement = instruction
        
        rsp = await self.react()

        self.set_todo(None)
        self.working_memory.clear()
        self.longterm_memory.clear()
        return rsp 
    
    async def _write_and_exec_code(self, max_retry: int = 3):
        counter = 0
        success = False
        
        if self.tool_recommender:
            # context = (self.working_memory.get()[-1].content if self.working_memory.get() else "")
            # tool_info = await self.tool_recommender.get_recommended_tool_info(context=context, plan=None)
            tool_info = ""
        else:
            tool_info = ""
            
        while not success and counter < max_retry:
            code, cause_by = await self._write_code(counter=counter, prompt_template_manager=self.prompt_template_manager, tool_info=tool_info, )
            
            self.working_memory.add(Message(content=code, role="assistant", cause_by=cause_by))
            
            result, success = await self.execute_code.run(code)
            
            self.working_memory.add(Message(content=result, role="user", cause_by=ExecuteNbCode))
            
            counter += 1
            
        return code, result, success
            
    async def _write_code(self, counter: int, prompt_template_manager: PromptTemplateManager, tool_info: str = ""):
        raise NotImplementedError("This method should be implemented in the subclass")