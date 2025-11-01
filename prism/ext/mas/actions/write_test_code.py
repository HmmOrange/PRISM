from typing import List

from tenacity import retry, wait_random_exponential, stop_after_attempt

from utils.logs import logger
from utils.common import CodeParser, general_after_log

from prism.prompts.prompt_template_manager import PromptTemplateManager
from prism.schema.message import Message
from prism.actions.action import Action

class WriteTestCode(Action):
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3), after=general_after_log(logger))
    async def run(self, user_requirement: str, model_information: str, prompt_template_manager: PromptTemplateManager, tool_info: str = "", use_reflection: bool = False,  longterm_memory: list[Message] = [],
                  working_memory: List[Message] = [], ):
        usr_input = prompt_template_manager.render(name="test_code", user_requirement=user_requirement, model_information=model_information)
        
        context = self.llm.format_msg(longterm_memory + usr_input + working_memory)
        
        rsp = await self.llm.aask(context)
        code = CodeParser.parse_code(text=rsp, lang="python")
        return code