import json
from pydantic import model_validator, Field
from copy import deepcopy
import tiktoken
from typing import Optional
from typing_extensions import Self

from prism.prompts.prompt_template_manager import PromptTemplateManager
from utils.common import CodeParser
from utils.logs import logger

from prism.hg_store.retriever import Retriever
from prism.schema.memory import Memory
from prism.schema.message import Message
from prism.utils.report import ThoughtReporter
from prism.actions.execute_nb_code import ExecuteNbCode 

from prism.settings import ml_coder_max_react_loop, retriever_top_k
from prism.ext.mas.agent.coder import Coder
from prism.ext.mas.actions.write_ml_code import WriteMLCode
from prism.ext.mas.actions.write_test_code import WriteTestCode

TEST_MODEL_REACT_THINK_PROMPT = """
# Your Role: Model Tester - Expert in Model Testing and Selection

You are a professional model testing specialist whose primary responsibility is to evaluate and select the best model for user requirements.

## Your Testing Workflow:
1. **Sample Data Inspection**: First, extract and print sample data to understand the data format and structure
2. **Initial Testing**: Run sample tests with 1 sample data to observe output from each model
3. **Output Mapping**: If model outputs are in format LABEL_0, LABEL_1, etc., map them to appropriate meaningful labels
4. **Comprehensive Testing**: Only if there are 2 or more models available, run full dataset on selected models to evaluate performance
5. **Final Selection**: Choose the model with the best performance

## Tooling Guidance:
- **Think explicitly about whether to combine tools to improve testing** (e.g., preprocessing, slicing, converting, or augmenting data) before/while running models.
- **Tools**: ['Terminal', 'AudioConverter', 'ImageResizer', 'ImageCropper', 'VideoFrameExtractor', 'VideoAudioExtractor', 'AudioSplitter', 'ImageSplitter', 'TextSplitter', 'TextTokenizer']


## Completion Criteria - Check if you have completed:
- **Sample Data Inspection**: Have you extracted and examined sample data to understand its structure?
- **Sample Testing**: Have you tested each available model with at least 1 sample data point?
- **Output Analysis**: Do you understand what kind of output each model produces?
- **Output Mapping**: If outputs are LABEL_0, LABEL_1 format, have you mapped them appropriately?
- **Full Dataset Testing**: If 2+ models available, have you run the selected model(s) with all available data/test cases?
- **Performance Evaluation**: Have you analyzed the results and can recommend the best model?
- **Tool Use Consideration**: Have you considered/used relevant tools to ensure fair, comparable inputs (e.g., consistent frame rate, tokenization, resizing)?

# User Requirement
{user_requirement}
# Context
{context}

Analyze your progress against the completion criteria above and decide the next step in your model testing workflow.

Output a json following the format:
```json
{{
    "thoughts": str = "Think step by step following the progress: (1) Have you inspected sample data? (2) Have you tested all models with sample data? (3) Have you mapped outputs if needed? (4) Do you have 2+ models for full dataset testing? (5) What testing still needs to be done? Based on this analysis, what's the next step?",
    "state": bool = "Return true if you still need to: inspect sample data, test more models with sample data, map outputs, or run comprehensive testing (if 2+ models available). Return false ONLY if you have completed all necessary steps including sample inspection, model testing, output mapping, and performance evaluation."
}}
```
"""

WRITE_CODE_REACT_THINK_PROMPT = """
# Your Role: AI Coder - Problem Solver and Solution Implementer

You are an AI coder responsible for implementing solutions to user requirements using available models, tools, and resources.

## Your Mission:
Implement and execute code to solve the user requirement completely, utilizing all available resources effectively.

## Available Resources to Consider:
1. **Models**: Use the provided models for inference and analysis
2. **Tools**: Leverage available tools for data preprocessing, manipulation, and enhancement
3. **Data**: Work with the provided data to achieve the goal
4. **Code Execution**: Run and test your implementations
5. **Experience Data**: Reuse prior results, processed variables, and partial solutions from earlier steps to improve performance and avoid redundant work
6. **Tools**: Maybe use need use tool to preprocess data.

## Completion Assessment Criteria:
Evaluate whether you have:

### **Requirement Fulfillment:**
- **Fully Solved**: The user requirement has been completely addressed
- **Partially Solved**: Some aspects remain unsolved but you've used all available resources
- **Not Solved**: Significant portions remain unsolved

### **Resource Utilization:**
- **Models Used**: Have you utilized the available models appropriately?
- **Tools Applied**: Have you used available tools for preprocessing, enhancement, etc.?
- **Data Processed**: Have you worked with the provided data effectively?
- **Code Executed**: Have you implemented and tested your solutions?
- **Experience Leveraged**: Have you reused prior steps and experience data to improve your current solution?
- **Tools**: ['Terminal', 'AudioConverter', 'ImageResizer', 'ImageCropper', 'VideoFrameExtractor', 'VideoAudioExtractor', 'AudioSplitter', 'ImageSplitter', 'TextSplitter', 'TextTokenizer']; 

### **Solution Quality:**
- **Working Implementation**: Does your code run without errors?
- **Reasonable Results**: Are the outputs meaningful and aligned with the requirement?
- **Resource Efficiency**: Have you used resources optimally?

## Decision Logic:
**Return true if:**
- User requirement is not fully solved AND you haven't exhausted all available resources
- You can still try different approaches with available models/tools
- There are reasonable next steps to attempt

**Return false if:**
- User requirement is completely solved, OR
- You have used ALL available resources (models, tools, approaches) and cannot solve the requirement further
- You've reached the practical limits of what's achievable with given resources

# User Requirement
{user_requirement}
# Context
{context}

Analyze your progress against the completion criteria and decide whether to continue or conclude.

Output a json following the format:
```json
{{
    "thoughts": str = "Analyze: (1) How much of the user requirement is solved? (2) What resources have you used (models, tools, data)? (3) What reasonable approaches remain? (4) Should you continue or conclude?",
    "state": bool = "Return true if you can still make meaningful progress with available resources. Return false if the requirement is solved OR you've exhausted all reasonable approaches with given resources."
}}
```
"""


class MLCoder(Coder):
    name: str = "ML Coder"
    profile: str = "A professional"
    model_type: str = ""
    model_information: str = ""
    model_testing_memory: Memory = Field(default_factory=Memory)
    max_react_loop: int = ml_coder_max_react_loop
    
    @model_validator(mode="after")
    def set_up(self) -> Self:
        self.retriever = Retriever()
        self.set_actions([WriteMLCode])
        self._set_state(0)
        return self
    
    def load_model_usage(self) -> str:
        path = "prism/ext/mas/templates/model.json"
        with open(path, "r", encoding="utf-8") as file:
            model_data = json.load(file)
            matched_data = model_data[self.model_type]
            interface = matched_data["interface"]
            output = matched_data["output"]
        return f"{self.model_type}. How to use:  with interface {interface} and output {output}."
    
    async def _think(self, test_model: bool) -> bool:
        """Useful in 'react' mode. Use LLM to decide whether and what to do next."""
        if test_model:
            context = self.model_testing_memory.get()
        else:
            context = self.working_memory.get()
        
        if not context:
            self.working_memory.add(self.get_memories()[0])
            self._set_state(0)
            return True
        think_logs = []
        counter = 0
        max_retry = 3
        while counter < max_retry:
            if test_model:
                prompt = TEST_MODEL_REACT_THINK_PROMPT.format(user_requirement=self.user_requirement, context=context)
            else:
                prompt = WRITE_CODE_REACT_THINK_PROMPT.format(user_requirement=self.user_requirement, context=context)
            try:
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
        if test_model:
            self.model_testing_memory.add(Message(content=rsp_dict["thoughts"], role="assistant"))
        else:
            self.working_memory.add(Message(content=rsp_dict["thoughts"], role="assistant"))
        need_action = rsp_dict["state"]
        self._set_state(0) if need_action else self._set_state(-1)
        
        return need_action
    
    async def _act(self, test_model: bool) -> Message:
        """Useful in 'react' mode. Return a Message conforming to Role._act interface."""
        code, _, _ = await self._write_and_exec_code(test_model = test_model)
        return Message(content=code, role="assistant", sent_from=self._setting, cause_by=self.todo)
    
    
    async def _react(self) -> Message:
        """Think first, then act, until the Role _think it is time to stop and requires no more todo.
        This is the standard think-act loop in the ReAct paper, which alternates thinking and acting in task solving, i.e. _think -> _act -> _think -> _act -> ...
        Use llm to select actions in _think dynamically
        """
        await self._retrieve_models()
        write_code_actions_taken = 0
        test_model_actions_taken = 0
        # while test_model_actions_taken < self.ac.max_react_loop:
        while test_model_actions_taken < 5:
            has_todo = await self._think(test_model=True)
            if not has_todo:
                break
            logger.debug(f"{self._setting}: {self.ac.state=}, will do {self.ac.todo}")
            await self._act(test_model=True)
            test_model_actions_taken += 1
        
        while write_code_actions_taken < self.ac.max_react_loop:
            # think
            has_todo = await self._think(test_model=False)
            if not has_todo:
                break
            # act
            logger.debug(f"{self._setting}: {self.ac.state=}, will do {self.ac.todo}")
            await self._act(test_model=False)
            write_code_actions_taken += 1
        code, output, success = await self._synthesize_code()
        return Message(content=code, role="assistant", metadata={
            "output": output,
            "success": success,
            "instruction": self.user_requirement,
            "previous_experience": deepcopy(self.experience)
        })

    def extract_model_type_from_request(self, query: str) -> Optional[str]:
        """
        Extract model type from user query if it contains exactly one of the supported task types.
        
        Args:
            query (str): User query text
            
        Returns:
            str or None: The detected model type, or None if not found or multiple types detected
        """
        # Define all supported model types
        MODEL_TYPES = [
            "token-classification", 
            "text-classification", 
            "zero-shot-classification", 
            "translation", 
            "summarization", 
            "question-answering", 
            "text-generation", 
            "sentence-similarity", 
            "tabular-classification", 
            "tabular-regression", 
            "object-detection", 
            "image-classification", 
            "image-to-text", 
            "automatic-speech-recognition", 
            "audio-classification", 
            "video-classification"
        ]
    
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        for model_type in MODEL_TYPES:
            if query_lower.find(model_type) != -1:
                return model_type
        
        return None
    
    async def _retrieve_models(self):
        model_type = self.extract_model_type_from_request(self.user_requirement)
        if not model_type:
            counter = 0
            max_retry = 3
            model_type_extraction_input = self.prompt_template_manager.render(name="extract_model_type", user_requirement=self.user_requirement)
            model_type_extraction_msg = self.llm.format_msg(model_type_extraction_input)
            model_type_error_logs = []
            while counter < max_retry:
                counter += 1
                try:
                    model_type_extraction_output = await self.llm.aask(model_type_extraction_msg + model_type_error_logs)
                    model_type_error_logs.append(Message(content=model_type_extraction_output, role="assistant"))
                    model_type = self.parse_xml(model_type_extraction_output, ["model_type"])["model_type"]
                    if model_type not in ["token-classification", "text-classification", "zero-shot-classification", "translation", "summarization", "question-answering", "text-generation", "sentence-similarity", "tabular-classification", "tabular-regression", "object-detection", "image-classification", "image-to-text", "automatic-speech-recognition", "audio-classification", "video-classification"]:
                        model_type_error_logs.append(Message(content="This model type is invalid!", role="user"))
                        continue
                    break
                except Exception as e:
                    model_type_error_logs.append(Message(content=str(e), role="user"))
                    continue
        if not model_type:
            raise Exception("Model type not found!")
        self.model_type = model_type
        documents = await self.retriever.retrieve(query=self.user_requirement, task_type=model_type, top_k=retriever_top_k)
        model_information = f"""To use model {self.model_type} please following instruction:
        {self.load_model_usage()}
        
        All available models you can use:
        """
        for idx, document in enumerate(documents):
            print(document["hash_id"])
            model_information += f"""{idx + 1}. {document["hash_id"]}
            Description: {document["content"]}
            """
        model_selection_input = self.prompt_template_manager.render(name="select_model", user_requirement=self.user_requirement, model_information = model_information)
        msg = self.llm.format_msg(model_selection_input)
        counter = 0
        max_retry = 3
        while counter < max_retry: 
            counter += 1
            try:
                model_selection_output = await self.llm.aask(msg)
                model_ids = json.loads(self.parse_xml(model_selection_output, ["model_ids"])["model_ids"])
                break
            except:
                continue
        model_ids = [model_id.strip().lower() for model_id in model_ids]
        if len(model_ids) > 0:
            new_documents = [document for document in documents if document["hash_id"].strip().lower() in model_ids]
        if len(model_ids) == 0 or len(new_documents) == 0:
            new_documents = documents[:5]
        self.model_information = f"""To use model {self.model_type} please following instruction:
        {self.load_model_usage()}
        
        All available models you can use:
        """
        for idx, document in enumerate(new_documents):
            self.model_information += f"""{idx + 1}. {document["hash_id"]}
            Hosted on: local
            Description: {document["content"]}
            """
    
    async def _write_and_exec_code(self, max_retry: int = 3, test_model: bool = False):
        counter = 0
        success = False
        
        if self.tool_recommender:
            # if test_model:
            #     context = (self.model_testing_memory.get()[-1].content if self.model_testing_memory.get() else "")
            # else:
            #     context = (self.working_memory.get()[-1].content if self.working_memory.get() else self.user_requirement)
            # tool_info = await self.tool_recommender.get_recommended_tool_info(context=context, plan=None)
            tool_info = ""
        else:
            tool_info = ""
            
        while not success and counter < max_retry:
            code, cause_by = await self._write_code(counter=counter, prompt_template_manager=self.prompt_template_manager, tool_info=tool_info, test_model = test_model)
            if test_model:
                self.model_testing_memory.add(Message(content=code, role="assistant", cause_by=cause_by))
            else:
                self.working_memory.add(Message(content=code, role="assistant", cause_by=cause_by))
            
            result, success = await self.execute_code.run(code)
            
            if test_model:
                self.model_testing_memory.add(Message(content=result, role="user", cause_by=ExecuteNbCode))
            else:
                self.working_memory.add(Message(content=result, role="user", cause_by=ExecuteNbCode))
            
            counter += 1
            
        return code, result, success
    
    async def _write_code(self, counter: int, prompt_template_manager: PromptTemplateManager, tool_info: str = "", test_model: bool = False):
        todo = self.ac.todo
        if not todo:
            raise Exception("No todo found!")
        logger.info(f"ready to {todo.name}")
        use_reflection = counter > 0 and self.use_reflection
        if test_model:
            code = await WriteTestCode().run(
                user_requirement=self.user_requirement,
                working_memory=self.model_testing_memory.get(),
                tool_info=tool_info,
                model_information=self.model_information,
                prompt_template_manager=prompt_template_manager,
                use_reflection=use_reflection,
                longterm_memory=self.longterm_memory.get(),
            )
            return code, WriteTestCode
        else:
            code = await WriteMLCode().run(
                user_requirement=self.user_requirement,
                working_memory=self.working_memory.get(),
                tool_info=tool_info,
                model_information=self.model_information,
                prompt_template_manager=prompt_template_manager,
                use_reflection=use_reflection,
                longterm_memory=self.longterm_memory.get(),
                model_testing_memory=self.model_testing_memory.get(),
            )
            return code, WriteMLCode