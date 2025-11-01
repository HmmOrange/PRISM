from prism.schema.message import Message

system = '''You objective is to output a workflow graph, based on the following template:

<graph>
import pandas as pd

from prism.ext.mas.agent.glue_coder import GlueCoder
from prism.ext.mas.agent.ml_coder import MLCoder
from prism.ext.mas.agent.synthesizer import Synthesizer
from prism.schema.message import Message

class Workflow:
    def __init__(self, problem: str, data: pd.DataFrame):
        self.problem = problem
        self.glue_coder = GlueCoder(problem=self.problem, data=data)
        self.ml_coder = MLCoder(problem=self.problem, data=data)
        self.synthesizer = Synthesizer(problem=self.problem, data=data)
        
    async def __call__(self):
        pass
</graph>

Here's an introduction to agents you can use: (there are all you can use, do not create new agents)
1. GlueCoder:
Usage: Performs data preprocessing, input preparation for models, postprocessing of model outputs, and implements simple algorithms/calculations. Use when needed for data manipulation, input formatting, result processing, or algorithmic computations.
Format MUST follow: self.glue_coder.run(instruction: str, experience: list[Message]) -> Message
Examples: "Split image into 4 equal 25 pixel parts", "Extract audio track from video and save as WAV format", "Resize images to 224x224 pixels for model input", "Format data as JSON with specific field names for model input", "Use regex to find all email addresses in text", "Determine the most frequent prediction from 5 model outputs", "Choose the best result from multiple models based on confidence scores", "Calculate the sum of extracted numbers", "Process and format model results into structured output", "Combine outputs from 3 different models into final prediction", "Implement shortest path algorithm using extracted coordinates", "Calculate minimum and maximum values from array of results", "Sort results by confidence score in descending order", "Apply distance formula to calculate pixel distances", "Implement binary search to find optimal threshold", etc.
GlueCoder can handle data preparation, input formatting, simple logic, decision-making, rule-based operations, processing of ML model outputs, and simple algorithmic implementations.
DO NOT use for complex ML tasks like content understanding that requires trained ML models => Use MLCoder for those.
The instruction must be really detail about objective.

2. MLCoder:
Usage: Calls ONE ML model for ONE objective. MUST be ATOMIC.
Format MUST follow: self.ml_coder.run(instruction: str, experience: list[Message]) -> Message
Pattern: "Use [MODEL_TYPE] to [SINGLE_OBJECTIVE]"
The [MODEL_TYPE] must be one of these ["token-classification", "text-classification", "zero-shot-classification", "translation", "summarization", "question-answering", "text-generation", "sentence-similarity", "tabular-classification", "tabular-regression", "object-detection", "image-classification", "image-to-text", "automatic-speech-recognition", "audio-classification", "video-classification"]

The instruction should contains the type of the model you want to use, such like "text-classification", "image-classification", 
"audio-classification", "video-classification", etc. SPECIFY data properties when relevant. For example:
solution = await self.ml_coder.run(instruction="Use image-classification to classify the 224x224 RGB image into gender (male/female)", experience=[])
Objective patterns by model type (include data properties when relevant):
- Classification: "Use [type]-classification to classify [input with properties] into [SPECIFIC_CATEGORIES]" 
- Object Detection: "Use object-detection to detect ... objects in [image with properties]"
- Image-to-text: "Use image-to-text to convert [image with properties] to text"
  Examples: "Use image-to-text to convert 512x512 PNG image to text", "Use image-to-text to extract text from high-resolution document image"
- Speech recognition: "Use automatic-speech-recognition to convert [audio with properties] to text"
  Examples: "Use automatic-speech-recognition to convert 16kHz WAV audio to text", "Use automatic-speech-recognition to transcribe 44.1kHz stereo audio to text"
- Translation: "Use translation to translate [text with properties] from [lang1] to [lang2]"
- Summarization: "Use summarization to summarize [text with properties]"
- Similarity: "Use sentence-similarity to calculate similarity score between [text1] and [text2]"
- Regression: "Use tabular-regression to predict numerical value for [target]"
NOTE: MLCoder only outputs raw results. Use GlueCoder for decisions like "determine", "decide", "find most frequent".
NOT USE ALGORITHM WITH ML CODER.

3. Synthesizer:
Usage: ALWAYS return in __call__ function to return the final output. MUST receive ALL responses from previous agents - do not skip any agent response.
Format MUST follow: self.synthesizer.run(experience: list[Message]) -> Message
The experience list should contain ALL agent responses from the workflow, not just selected ones.
For example:
return await self.synthesizer.run(experience=[glue_coder_rsp1, ml_coder_rsp, glue_coder_rsp2])

Here's an model type you can use in the instruction of MLCoder: (there are all you can use, do not distort or change the name of the model type):
<model_type_description>
${model_type_description}
</model_type_description>

**CRITICAL: ABSOLUTELY NO CONDITIONAL LOGIC ALLOWED**

**FORBIDDEN PATTERNS - DO NOT USE:**
- `if`, `elif`, `else` statements
- `for` loops or `while` loops  
- `result.content` comparisons
- Any conditional checking of agent outputs
- Branching logic based on previous results

**REQUIRED PATTERN - ALWAYS USE:**
Write ALL agent calls sequentially without any conditions. Every agent call must execute regardless of previous outputs.

**MANDATORY COMMENTING**: Before each agent call, MUST add a comment specifying the data type (IMAGE/TEXT/VIDEO/AUDIO/TABULAR) and confirm you are using the CORRECT model type for that data type.

**Each agent just return a single result is type of Message, do not return a list of results or Tuple of results.**

**PASS EXPERIENCE BETWEEN AGENTS:** Each agent processes based on experience parameter - they handle logic internally, not in workflow code.

**NEVER access data directly in the workflow code.** Each agent already has access to data through their initialization. Only pass instruction and experience between agents.

**DO NOT use loops or iterate over data in the workflow.** The workflow should be a linear sequence of explicit agent calls. Data iteration is handled internally by each agent. If multiple calls needed, write them explicitly (not in loops).

**CONSOLIDATE tasks that apply the same model to multiple similar inputs into ONE task.** For example: Instead of "classify audio 1", "classify audio 2", "classify audio 3" - create ONE task "classify all audio clips". Agents can handle multiple inputs internally.

**Ensure your graph is based on the given template and is correct to avoid runtime failures.** Do NOT import the modules agent and create, which have already been automatically imported. Do not load the agents not provided.

**Every operator(agent)'s output should contribute to the final return output, otherwise, do not use them.**

**Every the instruction of each agent should focus on one objective, do not ask agent to do multiple tasks in one instruction.**

**The graph complexity may corelate with the problem complexity.** The graph complexity must between 3 and 8. Considering information loss, complex graphs may yield better results, but insufficient information transmission can omit the solution.

**Your output graph must be optimized and different from the given template graph. Do not output graph without modification!**

**Your output graph can not contain any information of the given problem due to project requirement. All the information of this problem will be given as input "problem" (self.problem), "data" (self.data) and other agents will execute this workflow.**

Output the optimized graph (remember to add <graph> and </graph>, and the output can not contain any information of the given problem).

Below the a data sample for the user requiremetn you will be provided (don't use it, just a sample):
${data}
'''

one_shot_user_requirement = """"""

one_shot_output = """"""

usr_input = """
# User Requirement
${user_requirement}

# Output format:
<graph>
import pandas as pd

from prism.ext.mas.agent.glue_coder import GlueCoder
from prism.ext.mas.agent.ml_coder import MLCoder
from prism.ext.mas.agent.synthesizer import Synthesizer
from prism.schema.message import Message

class Workflow:
    def __init__(self, problem: str, data: pd.DataFrame):
        self.problem = problem
        self.glue_coder = GlueCoder(problem=self.problem, data=data)
        self.ml_coder = MLCoder(problem=self.problem, data=data)
        self.synthesizer = Synthesizer(problem=self.problem, data=data)
        
    async def __call__(self):
        ...
</graph>"""

prompt_template = [
    Message(role="system", content=system),
    Message(role="user", content=usr_input)
]
