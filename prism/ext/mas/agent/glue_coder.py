from pydantic import model_validator, BaseModel
from typing_extensions import Self


from prism.ext.mas.prompts.test_code import prompt_template
from prism.prompts.prompt_template_manager import PromptTemplateManager
from utils.logs import logger
from prism.settings import glue_coder_max_react_loop
from prism.ext.mas.agent.coder import Coder
from prism.ext.mas.actions.write_glue_code import WriteGlueCode

REACT_THINK_PROMPT = """
# Your Role: Glue Coder - Problem Solver and Solution Implementer

You are an AI coder responsible for implementing solutions to user requirements using glue codes, tools, and resources.

## Your Mission:
Implement and execute code to solve the user requirement completely, utilizing all available resources effectively.

## Your Implementation Workflow:
1. **Data Structure Inspection**: First, extract and examine the `data` variable to understand what you're working with
   - Print sample data: sample
   - For images: check dimensions, format, data type
   - For audio/video: check duration, format, file properties  
   - For text: check length, content, encoding
   - Don't make assumptions - inspect the actual data structure first
2. **Solution Implementation**: Based on data understanding, implement the solution
3. **Testing**: Verify that your solution works correctly with the actual data

## Available Resources to Consider:
1. **Code Implementation**: Write and execute code to solve the problem
2. **Tools**: Leverage available tools for data processing, manipulation, and analysis
3. **Data**: Work with the provided `data` variable - inspect it first before using
4. **Integration**: Connect different components and systems as needed
5. **Testing**: Verify that your solution works correctly

## Completion Assessment Criteria:
Evaluate whether you have:

### **Data Understanding:**
- **Data Inspected**: Have you examined the `data` variable structure and content?
- **Data Properties**: Do you understand the data format, dimensions, and properties?
- **No Assumptions**: Are you working with actual data structure, not assumptions?

### **Requirement Fulfillment:**
- **Fully Solved**: The user requirement has been completely addressed with working code
- **Partially Solved**: Some aspects are implemented but core functionality remains incomplete
- **Not Solved**: Significant portions of the requirement are still unaddressed

### **Implementation Quality:**
- **Working Code**: Does your implementation run without errors?
- **Correct Output**: Does the solution produce the expected results?
- **Complete Solution**: Have you addressed all aspects of the user requirement?
- **Proper Integration**: Are all components properly connected and working together?

### **Resource Utilization:**
- **Tools Applied**: Have you used available tools effectively for the task?
- **Data Processed**: Have you worked with the provided data appropriately?
- **Error Handling**: Have you handled potential errors and edge cases?

## Decision Logic:
**Return true if:**
- User requirement is not fully solved AND you can still make meaningful progress
- There are clear next steps to implement or improve the solution
- You haven't exhausted all reasonable approaches to solve the problem

**Return false if:**
- User requirement is completely solved with working implementation, OR
- You have tried all reasonable approaches and cannot solve the requirement further
- The solution is working correctly and meets all specified requirements

# User Requirement
{user_requirement}
# Context
{context}

Analyze your progress against the completion criteria and decide whether to continue implementing or conclude.

Output a json following the format:
```json
{{
    "thoughts": str = "Analyze: (1) Have you inspected the `data` variable structure first? (2) How much of the user requirement is solved? (3) Is your current implementation working correctly with the actual data? (4) What aspects still need to be addressed? (5) Can you make meaningful progress with next steps? Based on this analysis, should you continue or conclude?",
    "state": bool = "Return true if you can still make meaningful progress toward solving the user requirement. Return false if the requirement is fully solved with working implementation OR you've exhausted all reasonable approaches."
}}
```
"""

class GlueCoder(Coder):
    name: str = "Glue Coder"
    profile: str = "A professional coder"
    react_think_prompt: str = REACT_THINK_PROMPT
    max_react_loop: int = glue_coder_max_react_loop

    @model_validator(mode="after")
    def set_action_and_state(self) -> Self:
        self.set_actions([WriteGlueCode])
        self._set_state(0)
        return self
    
    async def _write_code(self, counter: int, prompt_template_manager: PromptTemplateManager, tool_info: str = ""):
        todo = self.ac.todo
        if not todo:
            raise Exception("no todo found!")
        logger.info(f"ready to {todo.name}")
        use_reflection = counter > 0 and self.use_reflection
        code = await todo.run(
            user_requirement=self.user_requirement,
            working_memory=self.working_memory.get(),
            tool_info=tool_info,
            prompt_template_manager=prompt_template_manager,
            use_reflection=use_reflection,
            longterm_memory=self.longterm_memory.get()
        )
        return code, todo