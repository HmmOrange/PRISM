from prism.schema.message import Message
from prism.prompts.templates.ner import one_shot_ner_paragraph, one_shot_ner_output

summarize_system = """Your task is to create a comprehensive, standardized description of machine learning models based on their original description and extracted entities.

You MUST:
1. Include ALL technical information present in the extracted entities
2. Write as a single, coherent paragraph in clear, factual, and detailed prose
3. Cover all aspects: architecture, training datasets, capabilities, frameworks, technical specifications
4. Organize information logically within the paragraph flow
5. Include specific details like class labels, resolution, number of classes, supported languages
6. Use consistent technical terminology
7. Write a comprehensive paragraph (4-8 sentences) that flows naturally

Include in your paragraph description:
- Model architecture and type
- Training and fine-tuning datasets with details
- Task capabilities and domain
- Technical specifications (resolution, number of classes, input/output formats)
- Supported languages
- Specific class labels or categories (if available)
- Input and output format requirements

DO NOT:
- Use bullet points, headers, or markdown formatting
- Add information not present in the entities
- Include author names, dates, or URLs
- Use promotional language
- Omit important technical details from the entities
- Break into multiple paragraphs or sections"""

example_input = f"""
Original passage:
```
{one_shot_ner_paragraph}
```

Extracted entities:
{one_shot_ner_output}
"""

example_output = """Vision Transformer (ViT) model is a transformer-based architecture designed for image classification tasks in the computer vision domain. The model is pre-trained on ImageNet-21k dataset (21,843 classes) and fine-tuned on ImageNet-2012/ImageNet-1k dataset (1,000 classes), processing images at 224x224 resolution. The model can classify images into 1,000 categories with specific examples including tiger, teapot, and palace among the classification labels, outputting results in class name format. This implementation includes tags for vision, transformers, and image-classification, indicating its primary use cases and technical specialization in computer vision tasks."""

user_input_template = """
Original passage:
```
${passage}
```

Extracted entities:
${named_entity_json}
"""

prompt_template = [
    Message(role="system", content=summarize_system),
    Message(role="user", content=example_input),
    Message(role="assistant", content=example_output),
    Message(role="user", content=user_input_template)
]