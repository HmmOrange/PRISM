from prism.schema.message import Message
from prism.prompts.templates.ner_query import query_prompt_one_shot_input, query_prompt_one_shot_output

hypothesis_system = """Your task is to create a comprehensive, standardized description of what the user is looking for based on their query and extracted entities.

You MUST:
1. Include ALL technical information present in the extracted entities
2. Write as a single, coherent paragraph in clear, factual, and detailed prose
3. Cover all aspects: desired architecture, training datasets, capabilities, frameworks, technical specifications
4. Organize information logically within the paragraph flow
5. Include specific details like class labels, resolution, number of classes, supported languages
6. Use consistent technical terminology
7. Write a comprehensive paragraph (4-8 sentences) that flows naturally

Include in your paragraph description:
- Desired model architecture and type
- Required training and fine-tuning datasets with details
- Task capabilities and domain the user wants
- Technical specifications (resolution, number of classes, input/output formats)
- Supported languages needed
- Specific class labels or categories required
- Input and output format requirements

DO NOT:
- Use bullet points, headers, or markdown formatting
- Add information not present in the entities
- Include author names, dates, or URLs
- Use promotional language
- Omit important technical details from the entities
- Break into multiple paragraphs or sections"""

example_input = f"""
User query:
```
{query_prompt_one_shot_input}
```

Extracted entities:
{query_prompt_one_shot_output}
"""

example_output = """The user is looking for an image classification model that can process digital images and categorize them into one of 1,000 ImageNet categories using computer vision techniques. The desired model should be capable of performing image-classification tasks in the vision domain, specifically trained on the ImageNet-1k dataset which contains 1,000 different object categories. The model should be able to classify input images and output results in a standardized format, making it suitable for general-purpose image recognition and object identification applications. This implementation would be ideal for users who need to automatically categorize and identify objects within digital images across a broad range of everyday objects and scenes."""

user_input_template = """
User query:
```
${query}
```

Extracted entities:
${named_entity_json}
"""

prompt_template = [
    Message(role="system", content=hypothesis_system),
    Message(role="user", content=example_input),
    Message(role="assistant", content=example_output),
    Message(role="user", content=user_input_template)
]