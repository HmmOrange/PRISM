from prism.schema.message import Message

system = """You are choosing the best-performing model for the user requirement.
Before give give your answers. you have to think step-by-step following pattern:

CRITICAL ANALYSIS REQUIREMENTS:
1. **Input Format Compatibility**: Carefully analyze the model description to understand:
   - What type of input the model expects (e.g., single-channel vs multi-channel images)
   - Any specific preprocessing requirements mentioned in the description
   - Whether the model is optimized for specific input characteristics

2. **Architecture-Specific Features**: Consider how different architectures handle inputs:
   - Some models are specifically designed for certain input formats
   - Architecture choice can significantly impact input compatibility
   - Look for mentions of input optimization or specialization

3. **Training Data Characteristics**: Examine what the model was trained on:
   - The original dataset characteristics (format, channels, etc.)
   - Any specific adaptations or modifications mentioned
   - How the training data format affects model expectations

4. **Model Description Clues**: Pay attention to:
   - Technical details about input processing
   - Mentions of specific input requirements
   - Any warnings or notes about input format
   - Architecture-specific input handling capabilities
   - Mentions of labels the model is trained on
   
5. **Postprocessing & Output Usability**:
   - Evaluate whether the model's output can be used to fulfill the task requirement either directly or with reliable transformation.
   - A model does not need to produce the final output in the exact target format, as long as its output is structured and clearly interpretable in a way that supports processing.
   - Acceptable transformations include:
     * Direct label mapping and translation
     * Mathematical operations on numeric outputs
     * Format conversions between different data types
     * Simple logical operations and rule-based classifications
   - Prefer models whose outputs align cleanly with task objectives, but also accept models that produce intermediate outputs which can be trivially or deterministically converted to the required form.
   - Models whose outputs are ambiguous, require complex interpretation, or depend on external assumptions for conversion should be avoided.
   Example: Label of model is LABEL_1, LABEL_2, LABEL_3, ... then you need to mapping to the correct labels or postprocessing to the correct format.

- **ENCOURAGED**: Select multiple models (2-5) if they are ALL truly relevant and fit the CRITICAL ANALYSIS REQUIREMENTS. This allows testing and comparison to find the best performing model.
- **CRITICAL**: Only select models that genuinely match the task requirements. DO NOT select irrelevant models or models with wrong labels just to increase the number of choices. 
- **QUALITY OVER QUANTITY**: Better to select 1-2 highly relevant models than 5 models where some are irrelevant or mismatched.
- **NOTE**: Maybe some models need to mapping to the correct labels or postprocessing to the correct format, that model can be highly rated.
- Choose the model that BEST MATCHES the specific requirements (number of classes, input format, output format) rather than choosing a "more powerful" model that doesn't fit the task. A model with more classes or different capabilities is NOT better if it doesn't match the exact needs of the task.
- At least you should have to choose 1 model.

You must response with the following XML structure:
<thought>
[str = "Thoughts really detail step by step following CRITICAL ANALYSIS REQUIREMENTS. For each candidate model, explicitly analyze whether it matches the task requirements before including it in the final selection. You have to analyze all the models. About 30 models. MUST NOT miss any model."]
</thought>
<model_ids>
[A list of the most promising model IDs from the provided list that best fits the task requirements, with A MAXIMUM OF 5 MODEL IDS. Must be an exact match from the available models, e.g. ["microsoft/DialoGPT-medium", ...]. Select multiple models ONLY if they are all truly relevant and suitable for the task. MUST RETURN A list of model_ids, not empty [].]
</model_ids>"""

one_shot_user_requirement = """
# User Requirement
I need to classify images into safe and unsafe content for content moderation. The dataset contains images that need to be categorized as either appropriate or inappropriate content.

# Available Models:
1. Falconsai/nsfw_image_detection - Vision Transformer (ViT) based on "google/vit-base-patch16-224-in21k", trained on ImageNet-21k dataset for NSFW image classification. Labels: "normal", "nsfw". Used for content safety and moderation.

2. microsoft/resnet-50 - ResNet-50 architecture trained on ImageNet-1k for general image classification. Labels: 1000 ImageNet classes (animals, objects, etc.). Used for general object recognition.

3. google/vit-base-patch16-224 - Vision Transformer for general image classification trained on ImageNet-21k. Labels: 21,000+ general categories. Used for broad image recognition tasks.

4. openai/clip-vit-base-patch32 - CLIP model for image-text understanding. Trained on image-text pairs. Labels: text-based descriptions. Used for image-text matching and zero-shot classification.
"""

one_shot_output = """<thought>
Let me analyze each model for safe/unsafe content classification:

1. Falconsai/nsfw_image_detection: Perfect match - designed for NSFW detection with exact "normal"/"nsfw" labels needed.
2. microsoft/resnet-50: Poor match - 1000 general ImageNet classes, not suitable for content safety.
3. google/vit-base-patch16-224: Poor match - 21,000+ general categories, too broad for binary safety classification.
4. openai/clip-vit-base-patch32: Moderate match - could work with prompts but not optimized for binary classification.

Best choice: Falconsai/nsfw_image_detection for exact task match.
</thought>
<model_ids>
["Falconsai/nsfw_image_detection", "openai/clip-vit-base-patch32"]
</model_ids>"""

usr_input = """
# User Requirement
${user_requirement}

# Available Models
${model_information}
"""

prompt_template = [
    Message(role="system", content=system),
    Message(role="user", content=one_shot_user_requirement),
    Message(role="assistant", content=one_shot_output),
    Message(role="user", content=usr_input)
]