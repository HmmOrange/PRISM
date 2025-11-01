from prism.schema.message import Message

system = '''You are tasked with creating a high-level sketch representation of a workflow graph to solve AI problems. This sketch will serve as an intermediate step before generating the actual code implementation.

Your task is to analyze the given problem and create a mermaid-style graph sketch that shows:
1. **Agents**: The processing units with their types and operations
2. **Data Flow**: How data moves between agents with proper typing
3. **Dependencies**: Which agents depend on outputs from other agents

**Available Agent Types:**
- **GlueCoder**: For ONE small step at a time - data preprocessing, postprocessing, simple algorithms, decision-making
  - Examples: "Split image into parts", "Calculate sum", "Apply threshold", "Format output"
  - NEVER combine multiple operations in one GlueCoder step
  Examples: "Split image into 4 equal 25 pixel parts", "Extract audio track from video and save as WAV format", "Resize images to 224x224 pixels for model input", "Format data as JSON with specific field names for model input", "Use regex to find all email addresses in text", "Determine the most frequent prediction from 5 model outputs", "Choose the best result from multiple models based on confidence scores", "Calculate the sum of extracted numbers", "Process and format model results into structured output", "Combine outputs from 3 different models into final prediction", "Implement shortest path algorithm using extracted coordinates", "Calculate minimum and maximum values from array of results", "Sort results by confidence score in descending order", "Apply distance formula to calculate pixel distances", "Implement binary search to find optimal threshold", etc.
GlueCoder can handle data preparation, input formatting, simple logic, decision-making, rule-based operations, processing of ML model outputs, and simple algorithmic implementations.
DO NOT use for complex ML tasks like content understanding that requires trained ML models => Use MLCoder for those.
  
- **MLCoder**: For calling ONE ML model for ONE specific objective (atomic operations)
  - Pattern: "Use [MODEL_TYPE] to [SINGLE_OBJECTIVE]"
  - Examples: "Use image-classification to classify into [specific categories]"
  - NEVER ask for multiple tasks in one MLCoder instruction
- **Synthesizer**: For final output aggregation (always the last step)

**Available ML Model Types:**
- Image: image-classification, image-to-text, object-detection
- Text: text-classification, token-classification, zero-shot-classification, translation, summarization, question-answering, text-generation, sentence-similarity
- Audio: audio-classification, automatic-speech-recognition
- Video: video-classification
- Tabular: tabular-classification, tabular-regression
- **Model Selection Guide**:
  - **Image Models**:
    + **image-to-text**: For descriptive captions, scene descriptions, complex content understanding
    + **image-classification**: For simple category recognition (dog/cat, happy/sad, single object identification)
    + **object-detection**: For locating multiple objects with positions, NOT for overall image description
  - **Text Models**:
    + **text-classification**: For categorizing entire text into predefined labels (sentiment, topic, intent)
    + **token-classification**: For word-level labeling (NER, POS tagging, entity extraction)
    + **zero-shot-classification**: For dynamic label sets without training data
    + **sentence-similarity**: For comparing text similarity, matching, relatedness
    + **text-generation**: For creating new content, explanations, code generation
    + **question-answering**: For extracting specific info from context text
    + **summarization**: For condensing long text while keeping key information
    + **translation**: For converting text between languages
  - **Audio Models**:
    + **audio-classification**: For direct audio categorization (language detection, genre, emotion, sound type) - EFFICIENT for classification tasks
    + **automatic-speech-recognition**: when you need the actual spoken text content for semantic analysis or text processing
  - **Tabular Models**:
    + **tabular-classification**: For predicting categories from structured data features
    + **tabular-regression**: For predicting continuous numerical values from structured data
  - **Video Models**:
    + **video-classification**: For categorizing videos as a whole (action types, sports, content categories)

**Data Type Specifications:**
- **Image**: Include properties like shape (e.g., 96x96, 224x224), format (PNG/JPG), quantity if multiple, etc.
- **Text**: Include properties like language, encoding, specific categories, etc.
- **Audio**: Include properties like sample rate, duration, format, etc.
- **Video**: Include properties like resolution, frame rate, duration, etc.
- **Tabular**: Include properties like columns, data types, etc.

**Important**:

Here's an model type you can use in the instruction of MLCoder: (there are all you can use, do not distort or change the name of the model type):
<model_type_description>
${model_type_description}
</model_type_description>

Here's a data sample for the user requiremetn you will be provided (don't use it, just a sample):
<data_sample>
${data}
</data_sample>

**Sketch Format (Mermaid-style):**
```mermaid
sketch
    %% Agents
    A1["AgentType|operation-description"]
    A2["AgentType|operation-description"]
    A3["AgentType|operation-description"]
    S["Synthesizer|aggregate"]

    %% Data Flows
    I1["DataType|properties"]
    D1["DataType|properties"]
    D2["DataType|properties"]
    D3["DataType|properties"]
    O1["DataType|properties"]

    %% Dependencies (experience relationships)
    I1 --> A1 --> D1
    D1 --> A2 --> D2
    D2 --> A3 --> D3
    D1, D2, D3 ---> S --> O1
```

**Critical Rules:**
1. **Atomic Operations**: Each agent = ONE small step, don't be greedy
   - GlueCoder: ONE operation per step (split, calculate, format, etc.)
   - MLCoder: ONE model type for ONE objective only
   - Break complex tasks into multiple simple steps
2. **Single Output Rule**: Each agent produces exactly ONE output data flow - NO multiple outputs (e.g., MUST NOT "I1 --> A1 --> D1, D2")
3. **Data Type Matching**: Ensure model types match input data types
4. **Experience Flow**: Show how experience passes between agents
5. **Consolidation**: Combine similar operations into one step instead of multiple separate steps
6. **Graph Complexity**: Use 3-8 agents total, matching problem complexity
7. **Algorithm**: Use GlueCoder to implement algorithms, NOT MLCoder
8. **MLCoder REQUIRED**: Include at least 1-2 MLCoder agents for better performance
9. **Content Extraction Rule**: To extract content from images or audio, MUST use MLCoder (image-to-text, automatic-speech-recognition), NOT GlueCoder
10. **Text-classification often better than zero-shot-classification**: Maybe try text-classification first if text-classification is available and working well else use zero-shot-classification because text-classification is specifically trained for a specific text classification task.
11. **Data Type Conversions**: Cross-modal conversions (Image→Text, Audio→Text, Video→Text) MUST use MLCoder, not GlueCoder
12. **Language Tasks**: Translation between languages and text summarization MUST use MLCoder with "translation" or "summarization" model types, not GlueCoder
13. **No multi-tasking**: Don't combine operations like "process AND analyze AND format" in one agent
14. **Synthesizer MUST receive ALL agent results**: Include every single response from all agents in experience list
15. **MLCoder operation-description MUST be exact model type**: Use ONLY these 16 model types: ["token-classification", "text-classification", "zero-shot-classification", "translation", "summarization", "question-answering", "text-generation", "sentence-similarity", "tabular-classification", "tabular-regression", "object-detection", "image-classification", "image-to-text", "automatic-speech-recognition", "audio-classification", "video-classification"]

**Your output should ONLY contain the sketch in the specified format, without any problem-specific details or explanations.**
'''

preprocess_one_shot_user_requirement = """A composite audio file contains 3 different music segments concatenated together. Each segment is 10 seconds long. Classify the genre of the middle segment."""

preprocess_one_shot_output = """```mermaid
sketch
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
    D1, D2 ---> S --> O1
```"""

one_shot_user_requirement = """Given a collection of product review texts, classify each review as positive, negative, or neutral sentiment."""

one_shot_output = """```mermaid
sketch
    %% Agents
    A1["MLCoder|text-classification"]
    S["Synthesizer|aggregate"]

    %% Data Flows
    I1["Text|type=product-reviews|language=English|quantity=multiple"]
    D1["Text|property=sentiment-labels|categories=[positive,negative,neutral]"]
    O1["Text|property=final-sentiment-predictions"]

    %% Dependencies
    I1 --> A1 --> D1
    D1 ---> S --> O1
```"""

postprocess_one_shot_user_requirement = """Analyze facial expressions in portrait photos and return "happy" if confidence score is above 0.8, otherwise return "uncertain"."""

postprocess_one_shot_output = """```mermaid
sketch
    %% Agents
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
    D1, D2 ---> S --> O1
```"""

usr_input = """
# User Requirement
${user_requirement}

# Output Format
```mermaid
```"""

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