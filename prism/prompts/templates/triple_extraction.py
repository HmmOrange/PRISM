from prism.prompts.templates.ner import one_shot_ner_paragraph, one_shot_ner_output
from prism.schema.message import Message

ner_conditioned_re_system = """Your task is to construct knowledge graph triples from Hugging Face model descriptions and their extracted entities.

You MUST respond with a JSON object that contains exactly this structure:
{
    "triples": [
        ["subject", "predicate", "object"],
        ...
    ]
}

Extract relationships that capture important model information using these predefined relationship types:
- Architecture: "is", "uses_architecture", "based_on"
- Training: "trained_on", "fine_tuned_on", "pre_trained_on"
- Framework: "supports", "compatible_with", "uses_framework"
- Tasks: "performs", "designed_for", "specializes_in"
- Domain: "belongs_to_domain", "focuses_on"
- Performance: "achieves", "has_accuracy", "has_metric"
- License: "licensed_under", "uses_license"
- Dataset: "trained_with", "evaluated_on", "uses_dataset"
- Technical: "has_resolution", "processes", "outputs", "classifies_into"

Focus on creating clear, factual triples. Use specific entities from the text as subjects and objects.
Always return the exact JSON structure with "triples" field containing array of 3-element arrays."""

ner_conditioned_re_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""

ner_conditioned_re_input = ner_conditioned_re_frame.format(passage=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output)

ner_conditioned_re_output = """
```json
{
    "triples": [
        ["Vision Transformer", "is", "transformer model"],
        ["ViT model", "uses_architecture", "vision transformer"],
        ["ViT model", "pre_trained_on", "ImageNet-21k"],
        ["ViT model", "fine_tuned_on", "ImageNet-2012"],
        ["ViT model", "performs", "image classification"],
        ["ViT model", "supports", "PyTorch"],
        ["ViT model", "supports", "TensorFlow"], 
        ["ViT model", "supports", "JAX"],
        ["ViT model", "licensed_under", "Apache 2.0"],
        ["ViT model", "has_resolution", "224x224"],
        ["ViT model", "classifies_into", "1000 classes"],
        ["ViT model", "belongs_to_domain", "vision"],
        ["ViT model", "classifies_into", "tiger"],
        ["ViT model", "classifies_into", "teapot"],
        ["ViT model", "classifies_into", "palace"],
        ["ImageNet-21k", "has_metric", "14 million images"],
        ["ImageNet-21k", "has_metric", "21843 classes"],
        ["ImageNet-2012", "has_metric", "1 million images"],
        ["ImageNet-2012", "has_metric", "1000 classes"]
    ]
}
```
"""

usr_input = """
Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
${passage}
```

${named_entity_json}
"""

prompt_template = [
    Message(role="system", content=ner_conditioned_re_system),
    Message(role="user", content=ner_conditioned_re_input),
    Message(role="assistant", content=ner_conditioned_re_output),
    Message(role="user", content=usr_input)
]