from prism.schema.message import Message

ner_system = """Your task is to extract model-related information from Hugging Face model card descriptions. 

You MUST respond with a JSON object that contains exactly these fields (no more, no less):
- "model_type": Array of model architectures (e.g., ["vit", "bert", "gpt2", "resnet", "t5", "roberta"])
- "task_type": Array of ML tasks from: ["token-classification", "text-classification", "zero-shot-classification", "translation", "summarization", "question-answering", "text-generation", "sentence-similarity", "tabular-classification", "tabular-regression", "object-detection", "image-classification", "image-to-text", "automatic-speech-recognition", "audio-classification", "video-classification"]
- "datasets": Array of training/fine-tuning datasets (e.g., ["imagenet-1k", "cifar100", "wikitext", "cc100", "squad"])
- "resolution": Array of input resolutions for vision/audio models (e.g., ["224x224", "16khz", "22050hz"])
- "classes": Array of classification labels (e.g., ["airplane", "cat", "fake news", "real news"])
- "language": Array of supported languages (e.g., ["english", "vietnamese", "multilingual", "en", "vi"])
- "domain": Array of application domains (e.g., ["vision", "nlp", "speech", "audio", "video", "tabular"])
- "num_classes": Array of number of classes (e.g., ["2", "1000", "21843"])
- "tags": Array of model tags/labels (e.g., ["pytorch", "vision", "transformers", "text-classification"])
- "input_format": Array of input format requirements (e.g., ["vi:", "en:", "text less than 500 words", "max_length 512"])
- "output_format": Array of output format details (e.g., ["LABEL_0", "LABEL_1", "class_names", "probabilities", "logits"])

For input_format field, extract:
- Required prefixes for translation models (e.g., "vi:", "en:")
- Text length limitations (e.g., "less than 500 words", "max_length 512")
- Special tokens or formatting requirements
- Audio/video input specifications (e.g., "16khz", "mono audio")
- Any specific input structure requirements

For output_format field, extract:
- Label format for classification models (e.g., "LABEL_0", "LABEL_1", "class_names")
- Output structure (e.g., "probabilities", "logits", "scores")
- Special output tokens or formatting
- Post-processing requirements (e.g., "argmax", "softmax")

If information for any field is not found, use an empty array []. Do not add extra fields.
Ignore author names, dates, URLs, and general descriptions. Focus only on technical specifications.
"""

one_shot_ner_paragraph = """Vision Transformer (ViT) model pre-trained on ImageNet-21k (14 million images, 21,843 classes) at resolution 224x224, and fine-tuned on ImageNet 2012 (1 million images, 1,000 classes) at resolution 224x224. 
This model supports image classification tasks and is compatible with PyTorch, TensorFlow, and JAX frameworks. 
The model uses vision transformer architecture and is licensed under Apache 2.0.
It can classify images into 1000 classes including categories like tiger, teapot, and palace."""


one_shot_ner_output = """
```json
{
    "model_type": ["vit", "vision transformer"],
    "task_type": ["image-classification"],
    "datasets": ["imagenet-21k", "imagenet-2012", "imagenet-1k"],
    "resolution": ["224x224"],
    "classes": ["tiger", "teapot", "palace"],
    "language": [],
    "domain": ["vision"],
    "num_classes": ["1000", "21843"],
    "tags": ["vision", "transformers", "image-classification"],
    "input_format": [],
    "output_format": ["class_names"]
}
```
"""

prompt_template = [
    Message(role="system", content=ner_system),
    Message(role="user", content=one_shot_ner_paragraph),
    Message(role="assistant", content=one_shot_ner_output),
    Message(role="user", content="${passage}")
]