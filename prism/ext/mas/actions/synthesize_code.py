from tenacity import retry, wait_random_exponential, stop_after_attempt

from utils.logs import logger
from utils.common import CodeParser, general_after_log

from prism.prompts.prompt_template_manager import PromptTemplateManager
from prism.schema.message import Message
from prism.actions.action import Action

class SynthesizeCode(Action):
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3), after=general_after_log(logger))
    async def run(self, user_requirement: str, working_memory: list[Message], prompt_template_manager: PromptTemplateManager, longterm_memory: list[Message] = []) -> str:
        
        write_code_input = prompt_template_manager.render(name="synthesize_code", user_requirement=user_requirement, context=working_memory)

        write_code_msg = self.llm.format_msg(longterm_memory + write_code_input)
        
        rsp = await self.llm.aask(write_code_msg)
        code = CodeParser.parse_code(text=rsp, lang="python")
        return code