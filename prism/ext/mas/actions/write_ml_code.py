from tenacity import retry, wait_random_exponential, stop_after_attempt

from utils.logs import logger
from utils.common import CodeParser, general_after_log

from prism.prompts.prompt_template_manager import PromptTemplateManager
from prism.schema.message import Message
from prism.actions.action import Action

class WriteMLCode(Action):

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3), after=general_after_log(logger))
    async def run(self, user_requirement: str, model_information: str, working_memory: list[Message], prompt_template_manager: PromptTemplateManager, use_reflection: bool = False,  tool_info: str = "", longterm_memory: list[Message] = [], model_testing_memory: list[Message] = []) -> str:
        
        write_code_input = prompt_template_manager.render(name="write_ml_code", user_requirement=user_requirement, model_information=model_information,  tool_info=tool_info)
        
        write_code_msg = self.llm.format_msg(longterm_memory + model_testing_memory + write_code_input + working_memory)
        rsp = await self.llm.aask(write_code_msg)
        code = CodeParser.parse_code(text=rsp, lang="python")
        return code