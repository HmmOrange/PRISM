from prism.schema.message import Message

system = '''Your ONLY task is to convert the provided sketch into executable workflow code. DO NOT think, analyze, or modify the design - just implement exactly what is shown.

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

Here's an introduction to agents you can use: 
1. GlueCoder:
Usage: Performs data preprocessing, input preparation for models, postprocessing of model outputs, and implements simple algorithms/calculations. Use when needed for data manipulation, input formatting, result processing, or algorithmic computations.
Format MUST follow: await self.glue_coder.run(instruction: str, experience: list[Message]) -> Message
The instruction must be really detail about objective.

2. MLCoder:
Usage: Calls ONE ML model for ONE objective. MUST be ATOMIC.
Format MUST follow: await self.ml_coder.run(instruction: str, experience: list[Message]) -> Message
Pattern: "Use [MODEL_TYPE] to [SINGLE_OBJECTIVE]"
The [MODEL_TYPE] must be one of these ["token-classification", "text-classification", "zero-shot-classification", "translation", "summarization", "question-answering", "text-generation", "sentence-similarity", "tabular-classification", "tabular-regression", "object-detection", "image-classification", "image-to-text", "automatic-speech-recognition", "audio-classification", "video-classification"]
The instruction must be really detail about the data properties.

3. Synthesizer:
Usage: ALWAYS return in __call__ function to return the final output. MUST receive ALL responses from previous agents - do not skip any agent response.
Format MUST follow: await self.synthesizer.run(experience: list[Message]) -> Message
**CRITICAL**: The experience list MUST contain ALL agent responses from the entire workflow, not just selected ones. Include every single agent result.

**CONVERSION RULES - NO THINKING REQUIRED:**
1. **Sketch agent A1 = Code line calling A1** (direct 1:1 mapping)
2. **Sketch dependencies = Code experience parameters** (A1 → A2 means A2 gets A1's result)
3. **ALL agents in sketch = ALL agents in code** (no exceptions, no reasoning)
4. **Synthesizer MUST receive ALL agent results** (every single response from all agents)
5. **ABSOLUTELY FORBIDDEN: if/else/for/while/any conditional or loop statements**
6. **ONLY communication method: experience parameter passing between agents**
7. **NO optimization comments** (no "this step is unnecessary" remarks)
8. **Every step in mermaid is meaningful**: Don't doubt any step
9. **STRICT SEQUENTIAL EXECUTION**: Each agent runs once, passes result via experience to next agent

**THIS IS PURE CONVERSION - NOT DESIGN TASK:**
- You are a sketch-to-code converter, not a workflow designer
- Implement exactly what is shown, do not improve or optimize
- No analysis, no reasoning, no justification - just convert
- ZERO TOLERANCE for control flow statements (if/else/for/while/try/except)
- ALL data flow MUST happen through experience parameter only

# Output Format
```python
your code
```'''

preprocess_one_shot_user_requirement = """
<information>
    <user_requirement>A composite audio file contains 3 different music segments concatenated together. Each segment is 10 seconds long. Classify the genre of the middle segment.</user_requirement>
    <sketch>sketch
    %% Agents
    A1["GlueCoder|extract-middle-segment"]
    A2["MLCoder|audio-classification"]
    S["Synthesizer|aggregate"]

    %% Data Flows
    I1["Audio|duration=30s|segments=3|format=WAV"]
    D1["Audio|duration=10s|segment=middle|format=WAV"]
    D2["Text|property=genre-label"]
    O1["Text|property=final-genre-prediction"]

    %% Dependencies
    I1 --> A1 --> D1
    D1 --> A2 --> D2
    A1, A2 ---> S --> O1</sketch>
</information>
"""

preprocess_one_shot_output = '''```python
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
        middle_segment = await self.glue_coder.run(instruction="Extract the middle 10-second segment from the 30-second composite audio file", experience=[])
        
        genre_prediction = await self.ml_coder.run(instruction="Use audio-classification to classify the extracted audio segment into music genre categories", experience=[middle_segment])
        
        return await self.synthesizer.run(experience=[middle_segment, genre_prediction])
```'''

one_shot_user_requirement = """
<information>
    <user_requirement>Given a collection of product review texts, classify each review as positive, negative, or neutral sentiment.</user_requirement>
    <sketch>sketch
    %% Agents
    A1["MLCoder|text-classification"]
    S["Synthesizer|aggregate"]

    %% Data Flows
    I1["Text|type=product-reviews|language=English|quantity=multiple"]
    D1["Text|property=sentiment-labels|categories=[positive,negative,neutral]"]
    O1["Text|property=final-sentiment-predictions"]

    %% Dependencies
    I1 --> A1 --> D1
    A1 ---> S --> O1</sketch>
</information>
"""

one_shot_output = '''```python
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
        sentiment_predictions = await self.ml_coder.run(instruction="Use text-classification to classify each product review into positive, negative, or neutral sentiment categories", experience=[])
        
        return await self.synthesizer.run(experience=[sentiment_predictions])
```'''

postprocess_one_shot_user_requirement = """
<information>
    <user_requirement>Analyze facial expressions in portrait photos and return "happy" if confidence score is above 0.8, otherwise return "uncertain".</user_requirement>
    <sketch>%% Agents
    A1["MLCoder|image-classification"]
    A2["GlueCoder|confidence-threshold-decision"]
    S["Synthesizer|aggregate"]

    %% Data Flows
    I1["Image|type=portrait-photos|format=RGB|resolution=variable"]
    D1["Text|property=expression-labels|confidence=scores"]
    D2["Text|property=binary-decision|categories=[happy,uncertain]|threshold=0.8"]
    O1["Text|property=final-expression-result"]

    %% Dependencies
    I1 --> A1 --> D1
    D1 --> A2 --> D2
    A1, A2 ---> S --> O1</sketch>
</information>"""

postprocess_one_shot_output = '''```python
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
        expression_analysis = await self.ml_coder.run(instruction="Use image-classification to analyze facial expressions in portrait photos and return expression labels with confidence scores", experience=[])
        
        threshold_decision = await self.glue_coder.run(instruction="Apply confidence threshold logic: return 'happy' if confidence score is above 0.8, otherwise return 'uncertain'", experience=[expression_analysis])
        
        return await self.synthesizer.run(experience=[expression_analysis, threshold_decision])
```'''


usr_input = '''
<information>
    <user_requirement>${user_requirement}</user_requirement>
    <sketch>${sketch}</sketch>
</information>

Your output should ONLY contain the workflow code in the specified format, without any problem-specific details or explanations.
```python
```
'''

prompt_template = [
    Message(role="system", content=system),
    Message(role="user", content=preprocess_one_shot_user_requirement),
    Message(role="assistant", content=preprocess_one_shot_output),
    Message(role="user", content=one_shot_user_requirement),
    Message(role="assistant", content=one_shot_output),
    Message(role="user", content=postprocess_one_shot_user_requirement),
    Message(role="assistant", content=postprocess_one_shot_output),
    Message(role="user", content=usr_input)
]